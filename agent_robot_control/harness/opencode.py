"""Launch OpenCode headless against the robot MCP server.

Config generation is unit-tested; the end-to-end launch is NOT exercised until
an API key is provided (PLAN.md Section 8). Flags follow the OpenCode CLI as
of 2026-09 (`opencode run [message]`, `--model provider/model`,
`--format json`); verify against `opencode run --help` before the first run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig

from agent_robot_control.harness.base import HarnessOutcome, \
    prepare_initial_observation, render_task_prompt, run_process, \
    server_command, server_env, system_prompt_text, which_or_none


def write_opencode_config(cfg: DictConfig, run_dir: Path, workspace: Path,
                          system_prompt_file: Path) -> Path:
    """`opencode.json` in the workspace (OpenCode reads it from cwd)."""
    conf: Dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "model": str(cfg.harness.model),
        "instructions": [str(system_prompt_file)],
        "mcp": {
            "robot": {
                "type": "local",
                "command": server_command(run_dir),
                "enabled": True,
                "environment": server_env(),
                "timeout": int(cfg.harness.mcp_tool_timeout_ms),
            }
        },
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
    }
    path = workspace / "opencode.json"
    path.write_text(json.dumps(conf, indent=2))
    return path


def build_command(cfg: DictConfig, prompt: str) -> List[str]:
    """The `opencode run` command line."""
    h = cfg.harness
    return [which_or_none(h.binary) or h.binary, "run", "--format", "json",
            "--model", str(h.model), prompt]


def parse_json_transcript(path: Path) -> Dict[str, Any]:
    """Best-effort parse of OpenCode's JSON event stream."""
    turns = 0
    tool_calls: Dict[str, int] = {}
    final_text = ""
    md = ["# Transcript (opencode)\n"]
    for line in path.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        et = str(ev.get("type", ""))
        if "tool" in et:
            name = str(ev.get("name") or ev.get("tool") or ev.get("part", {}).get("tool", "?"))
            tool_calls[name] = tool_calls.get(name, 0) + 1
            md.append(f"**Tool call `{name}`**\n")
        elif "text" in et:
            turns += 1
            text = str(ev.get("text") or ev.get("part", {}).get("text", ""))
            final_text = text or final_text
            md.append(f"**Assistant:** {text}\n")
    return {"turns": turns, "tool_calls": tool_calls, "final_text": final_text,
            "markdown": "\n".join(md)}


def run_opencode(cfg: DictConfig, run_dir: Path, tool_names: List[str]) -> HarnessOutcome:
    """Full launch (untested end to end until an API key is available)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    initial_image, robot_state = prepare_initial_observation(cfg, run_dir)
    prompt = render_task_prompt(cfg, run_dir, initial_image, robot_state, tool_names)
    (run_dir / "prompt.md").write_text(prompt)
    sys_file = run_dir / "system_prompt.md"
    sys_file.write_text(system_prompt_text())
    write_opencode_config(cfg, run_dir, workspace, sys_file)
    cmd = build_command(cfg, prompt)
    (run_dir / "harness_command.json").write_text(json.dumps(cmd, indent=2))
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    transcript = run_dir / "transcript.jsonl"
    code, wall, timed_out = run_process(cmd, workspace, env, transcript,
                                        run_dir / "harness_stderr.log",
                                        float(cfg.harness.wall_clock_limit_s))
    parsed = parse_json_transcript(transcript) if transcript.exists() else {}
    if parsed:
        (run_dir / "transcript.md").write_text(parsed["markdown"])
    return HarnessOutcome(exit_code=code, wall_time_s=wall, timed_out=timed_out,
                          num_assistant_turns=parsed.get("turns", 0),
                          num_tool_calls=sum(parsed.get("tool_calls", {}).values()),
                          tool_call_counts=parsed.get("tool_calls", {}),
                          final_text=parsed.get("final_text", ""))
