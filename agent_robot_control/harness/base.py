"""Common harness plumbing: prompt rendering, MCP config, launch + logging."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[2]
PROMPTS = Path(__file__).resolve().parent / "prompts"


@dataclass
class HarnessOutcome:
    """What the runner learns from one harness session."""
    exit_code: int
    wall_time_s: float
    timed_out: bool
    num_assistant_turns: int = 0
    num_tool_calls: int = 0
    tool_call_counts: Dict[str, int] = field(default_factory=dict)
    final_text: str = ""
    cost_usd: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def server_command(run_dir: Path) -> List[str]:
    """Command that starts the stdio MCP server for ``run_dir``.

    The interpreter is named directly rather than going through ``uv run``.
    ``uv run`` takes a lock on the shared ``.venv`` while it revalidates the
    environment, so several array tasks starting at once serialise on it: on
    2026-09-09 two of three domino runs sat behind that lock until Claude
    Code's 300 s MCP connect timeout fired, and their agents spent the run
    reporting that no robot tools existed. ``sys.executable`` is already the
    project venv's python (run_experiment itself runs under ``uv run``), and
    a plain interpreter start needs no lock.
    """
    python = sys.executable or "python"
    if not Path(python).exists():  # pragma: no cover - defensive
        python = str(REPO / ".venv" / "bin" / "python")
    inner = " ".join(shlex.quote(a) for a in [
        python, "-m", "agent_robot_control.mcp_server.server",
        "--run-dir", str(run_dir)
    ])
    # Keep the server's stderr: the harness pipes it somewhere we cannot read,
    # so a server that fails to start used to leave no trace at all - the agent
    # just reported that no robot tools existed (2026-09-09).
    log = shlex.quote(str(run_dir / "mcp_server.log"))
    # Run from the repo. The server would otherwise inherit the harness's cwd,
    # which is the agent's workspace sandbox, and env settings that name a
    # RELATIVE path then resolve inside that sandbox. The domino min-block task
    # cache (CFG.domino_min_block_task_cache_dir = "saved_datasets/...") is one:
    # every server missed the repo's cache and regenerated its task, 217-345 s
    # of search against Claude Code's 300 s MCP connect timeout, so two of
    # three domino-turn runs started with no robot tools at all (2026-09-09).
    return ["bash", "-c",
            f"cd {shlex.quote(str(REPO))} && exec {inner} 2>> {log}"]


def server_env() -> Dict[str, str]:
    """Environment for the server subprocess."""
    return {"OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            # ``uv run`` used to put the repo on the path; naming the
            # interpreter directly means saying so here instead.
            "PYTHONPATH": str(REPO)}


def render_task_prompt(cfg: DictConfig, run_dir: Path, initial_image: Path,
                       robot_state: str, tool_names: List[str]) -> str:
    """Fill the task prompt template."""
    template = (PROMPTS / "task_template.md").read_text()
    tool_list = "\n".join(f"- `{t}`" for t in tool_names)
    return template.format(task_text=cfg.env.task_text.strip(),
                           initial_image=str(initial_image),
                           robot_state=robot_state,
                           interaction_cap=int(cfg.server.interaction_cap),
                           workspace=str(run_dir / "workspace"),
                           tool_list=tool_list)


def system_prompt_text() -> str:
    """Shared system prompt for both harnesses."""
    return (PROMPTS / "system.md").read_text()


def prepare_initial_observation(cfg: DictConfig, run_dir: Path):
    """Render the initial camera image and robot state with a throwaway
    session (same seed/task as the server will build), so the prompt can
    reference them before the server exists. Returns (image_path, text)."""
    from agent_robot_control.mcp_server.server import build_session
    ws = run_dir / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    tmp_cfg = OmegaConf.merge(cfg, {"run_dir": None,
                                    "server": {"log_transitions": False}})
    session = build_session(tmp_cfg)
    try:
        import imageio.v2 as imageio
        path = ws / "initial.png"
        imageio.imwrite(path, session.render())
        text = "Robot now: " + session.robot_state_text()
    finally:
        import pybullet as p
        p.disconnect(physicsClientId=session.env._physics_client_id)
    return path, text


def run_process(cmd: List[str], cwd: Path, env: Dict[str, str],
                stdout_path: Path, stderr_path: Path, timeout_s: float,
                stdin_text: Optional[str] = None) -> tuple:
    """Run ``cmd`` streaming stdout/stderr to files. Returns
    (exit_code, wall_time, timed_out)."""
    t0 = time.time()
    timed_out = False
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=out,
                                stderr=err,
                                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL)
        try:
            if stdin_text is not None:
                proc.stdin.write(stdin_text.encode())  # type: ignore[union-attr]
                proc.stdin.close()  # type: ignore[union-attr]
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return proc.returncode, time.time() - t0, timed_out


def which_or_none(binary: str) -> Optional[str]:
    """Absolute path of ``binary`` or None."""
    return shutil.which(binary)
