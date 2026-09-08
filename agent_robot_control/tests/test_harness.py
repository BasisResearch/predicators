"""Harness config/prompt generation (no LLM calls)."""
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from agent_robot_control.harness import claude_code, opencode
from agent_robot_control.harness.base import render_task_prompt, server_command

CONF = str(Path(__file__).resolve().parents[1] / "conf")


def _cfg(tmp_path, *overrides):
    with initialize_config_dir(version_base=None, config_dir=CONF):
        cfg = compose(config_name="config",
                      overrides=[f"output_root={tmp_path}", *overrides])
    return cfg


def test_config_composes_all_groups(tmp_path):
    for env in ["airport", "donut", "plug_outlet"]:
        for cond in ["move_to", "model_free", "model_based"]:
            cfg = _cfg(tmp_path, f"env={env}", f"condition={cond}", "seed=2")
            assert cfg.env.name.startswith("pybullet_")
            assert "move_to" in cfg.condition.tools
            tier = cfg.env.get("clearance_tier", "") or ""
            assert str(cfg.run_dir).endswith(f"{cfg.env.name}{tier}/{cond}/claude_code/seed_2")
    cfg = _cfg(tmp_path, "rl=ppo")
    assert cfg.rl.algo == "ppo"


def test_claude_code_command_and_mcp_config(tmp_path):
    cfg = _cfg(tmp_path, "env=donut", "condition=model_free")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mcp = claude_code.write_mcp_config(run_dir)
    data = json.loads(mcp.read_text())
    srv = data["mcpServers"]["robot"]
    assert srv["args"][-2:] == ["--run-dir", str(run_dir)]
    assert srv["env"]["OPENBLAS_NUM_THREADS"] == "1"
    sys_file = run_dir / "system.md"
    sys_file.write_text("x")
    cmd = claude_code.build_command(cfg, run_dir, "do it", mcp, sys_file,
                                    list(cfg.condition.tools))
    joined = " ".join(cmd)
    assert "--strict-mcp-config" in joined and "--output-format stream-json" in joined
    assert "mcp__robot__run_rl_on_particles" in joined
    assert "mcp__robot__run_model_based_rl_on_particles" not in joined
    assert "--model claude-sonnet-5" in joined


def test_opencode_config(tmp_path):
    cfg = _cfg(tmp_path, "harness=opencode", "condition=model_based")
    run_dir = tmp_path / "run"
    ws = run_dir / "workspace"
    ws.mkdir(parents=True)
    sys_file = run_dir / "system.md"
    sys_file.write_text("x")
    path = opencode.write_opencode_config(cfg, run_dir, ws, sys_file)
    data = json.loads(path.read_text())
    assert data["mcp"]["robot"]["command"] == server_command(run_dir)
    assert data["model"] == "anthropic/claude-sonnet-5"


def test_prompt_renders_tools_and_budget(tmp_path):
    cfg = _cfg(tmp_path, "env=airport", "condition=move_to", "server.interaction_cap=1234")
    text = render_task_prompt(cfg, tmp_path, tmp_path / "initial.png", "Robot now: ok",
                              list(cfg.condition.tools))
    assert "item_2" in text and "1234" in text and "`move_to`" in text and "`wait`" in text
    assert "run_rl_on_particles" not in text


def test_stream_json_parser(tmp_path):
    lines = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "hi"},
                                                       {"type": "tool_use", "name": "mcp__robot__move_to", "input": {"x": 1}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": [{"type": "text", "text": "ok"}]}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "subtype": "success", "num_turns": 2, "total_cost_usd": 0.5, "result": "done"},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines))
    parsed = claude_code.parse_stream_json(p)
    assert parsed["turns"] == 2 and parsed["tool_calls"] == {"mcp__robot__move_to": 1}
    assert parsed["cost_usd"] == 0.5 and parsed["final_text"] == "done"
    assert "Tool call" in parsed["markdown"]
