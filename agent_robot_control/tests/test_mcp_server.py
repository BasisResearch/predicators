"""Drive the stdio MCP server with a real MCP client subprocess."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]


def _write_config(run_dir: Path, tools):
    cfg = OmegaConf.create({
        "seed": 0,
        "env": {"name": "pybullet_donut", "task_idx": 0},
        "server": {"interaction_cap": 400, "camera_width": 224,
                   "camera_height": 126, "particles_per_object": 16,
                   "log_transitions": False},
        "condition": {"tools": tools},
        "rl": {"algo": "sac", "points_per_object": 8,
               "algo_kwargs": {"learning_starts": 10, "batch_size": 8},
               "early_stop_successes": 1, "early_stop_window": 1,
               "final_exec_attempts": 1},
        "run_dir": str(run_dir),
    })
    OmegaConf.save(cfg, run_dir / "config.yaml")


async def _drive(run_dir: Path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    env = dict(os.environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_robot_control.mcp_server.server", "--run-dir", str(run_dir)],
        env=env, cwd=str(REPO))
    out = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            out["tools"] = sorted(t.name for t in tools.tools)
            r = await session.call_tool("pixels_to_particles", {"max_points_per_object": 8})
            out["particles"] = r
            r = await session.call_tool("move_to", {"x": 1.4, "y": 0.6, "z": 0.35, "gripper": "open", "max_steps": 40})
            out["move"] = r
            r = await session.call_tool("wait", {"steps": 7})
            out["wait"] = r
            r = await session.call_tool("run_rl_on_particles", {
                "reward_code": "def reward(p, v, e, q, g):\n    return -float(np.linalg.norm(e[:2] - p['target'].mean(0)[:2])) if len(p['target']) else -1.0",
                "budget_interactions": 30, "episode_length": 5})
            out["rl"] = r
            r = await session.call_tool("run_rl_on_particles", {"reward_code": "def nope(): pass", "budget_interactions": 30})
            out["bad_reward"] = r
    return out


def test_server_end_to_end(tmp_path):
    _write_config(tmp_path, ["move_to", "wait", "pixels_to_particles", "run_rl_on_particles"])
    out = asyncio.run(_drive(tmp_path))
    assert out["tools"] == ["move_to", "pixels_to_particles", "run_rl_on_particles", "wait"]
    assert "Waited 7 env steps" in out["wait"].content[0].text
    types = [c.type for c in out["particles"].content]
    assert "text" in types and "image" in types
    text = out["particles"].content[0].text
    assert "donut_0" in text and "particles_0000.npz" in text
    assert (tmp_path / "workspace" / "particles_0000.npz").exists()
    mtext = out["move"].content[0].text
    assert "move_to" in mtext and "interactions used" in mtext
    rtext = out["rl"].content[0].text
    assert "RL finished" in rtext, rtext
    btext = out["bad_reward"].content[0].text
    assert "ERROR" in btext and "reward" in btext
    events = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert "tool_call" in kinds and "rl_done" in kinds
    assert (tmp_path / "results.json").exists()
