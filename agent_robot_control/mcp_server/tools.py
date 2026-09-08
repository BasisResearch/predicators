"""MCP tool implementations. Each returns a list of content blocks
(text + current camera image). The server wires them to a ``SimSession``."""
from __future__ import annotations

import io
import json
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
from mcp.server.mcpserver import Image

from agent_robot_control.rl.backend import RLRequest
from agent_robot_control.rl.reward_loader import REWARD_SIGNATURE, RewardError, \
    load_reward, validate_reward
from agent_robot_control.sim.budget import BudgetExhausted
from agent_robot_control.sim.ee_control import GRIPPER_COMMANDS

MAX_TEXT = 30_000

REWARD_CONTRACT = f"""Reward code contract:
{REWARD_SIGNATURE}
  particles: dict name -> (K,3) float array of world xyz points for that object
             (K may be 0 if the object is occluded / out of view).
  visible:   dict name -> (K,) bool array (all True for present points).
  ee_pos:    (3,) end-effector position; ee_quat: (4,) xyzw quaternion.
  gripper:   finger joint value, 0.04 = fully open (~8 cm) ... 0.01 = closed (~2 cm).
  Return a float. A value >= 1.0 means the goal is accomplished (ends the
  episode as a success and drives early stopping). Smaller values shape
  progress; keep them roughly in [-10, 1]. Only numpy (np) and math are
  available; no imports."""


class ToolContext:
    """Shared state for all tools: the session, RL backends, a lock."""

    def __init__(self, session, backends: Dict[str, Any], rl_defaults: Dict[str, Any]) -> None:
        self.session = session
        self.backends = backends
        self.rl_defaults = rl_defaults
        self.lock = threading.Lock()
        self.tool_calls = 0

    # ── Result helpers ──────────────────────────────────────────

    def image_block(self, tag: str) -> Image:
        """Save the current camera image to the workspace and wrap it."""
        path = self.session.snapshot_image(tag)
        if path is not None:
            return Image(path=path)
        buf = io.BytesIO()
        imageio.imwrite(buf, self.session.render(), format="png")
        return Image(data=buf.getvalue(), format="png")

    def status_lines(self) -> str:
        s = self.session
        return (f"Robot: {s.robot_state_text()}\n"
                f"Budget: {s.budget.status()}.")

    def spill(self, text: str, tag: str) -> str:
        if len(text) <= MAX_TEXT or self.session.workspace is None:
            return text
        path = self.session.workspace / f"{tag}_output.txt"
        path.write_text(text)
        return text[:MAX_TEXT] + f"\n... [truncated; full text in {path}]"

    def result(self, text: str, tag: str, with_image: bool = True) -> List[Any]:
        blocks: List[Any] = [self.spill(text + "\n" + self.status_lines(), tag)]
        if with_image:
            img = self.image_block(tag)
            blocks.append(img)
            if img.path is not None:
                blocks[0] += f"\nCurrent camera image attached (also saved to {img.path})."
        return blocks

    def error(self, text: str, tag: str) -> List[Any]:
        return self.result("ERROR: " + text, tag)

    def log_call(self, tool: str, args: Dict[str, Any], before: int) -> None:
        self.tool_calls += 1
        self.session.record_event("tool_call", tool=tool, args=args,
                                  interactions_before=before,
                                  interactions_after=self.session.interactions,
                                  call_index=self.tool_calls)
        self.session.write_results()


# ── Tools ─────────────────────────────────────────────────────

def move_to(ctx: ToolContext, x: float, y: float, z: float,
            roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
            gripper: str = "keep", max_steps: int = 200) -> List[Any]:
    """Implementation of the move_to tool."""
    s = ctx.session
    before = s.interactions
    if gripper not in GRIPPER_COMMANDS:
        return ctx.error(f"gripper must be one of {GRIPPER_COMMANDS}.", "move_to")
    if s.budget.exhausted:
        return ctx.error("Interaction budget exhausted; no more actions possible.", "move_to")
    quat = s.controller.quat_from_rpy_deg(roll, pitch, yaw)
    try:
        res = s.controller.move_to((x, y, z), quat, gripper=gripper,
                                   max_steps=int(max_steps))
        text = "move_to " + res.summary()
    except BudgetExhausted:
        text = "move_to stopped: the interaction budget is exhausted."
    ctx.log_call("move_to", dict(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                 gripper=gripper, max_steps=max_steps), before)
    return ctx.result(text, "move_to")


def wait(ctx: ToolContext, steps: int = 50) -> List[Any]:
    """Hold the arm still for ``steps`` env steps (time passes; belts move)."""
    s = ctx.session
    before = s.interactions
    n = int(np.clip(steps, 1, 2000))
    if s.budget.exhausted:
        return ctx.error("Interaction budget exhausted; no more actions possible.", "wait")
    done = 0
    try:
        for _ in range(n):
            s.step(s.controller.hold_action())
            done += 1
    except BudgetExhausted:
        pass
    ctx.log_call("wait", dict(steps=n), before)
    return ctx.result(f"Waited {done} env steps holding the current pose.", "wait")


def pixels_to_particles(ctx: ToolContext, max_points_per_object: int = 64) -> List[Any]:
    """Implementation of the pixels_to_particles tool."""
    s = ctx.session
    before = s.interactions
    k = int(np.clip(max_points_per_object, 4, 512))
    snap, summary, npz, js = s.save_particles(k)
    lines = [
        f"Extracted particles for {len(snap.names)} named objects "
        f"({k} points max per object).",
        f"Files: {npz} (arrays points_<name> (K,3), colors_<name> (K,3), "
        f"ee_pos, ee_quat_xyzw, gripper) and {js} (this summary).",
        "Object summary (world coordinates, metres):",
        json.dumps(summary["objects"], indent=1),
    ]
    if summary.get("unlabeled_body_ids"):
        lines.append("Unlabeled bodies (floor/walls) were skipped.")
    ctx.log_call("pixels_to_particles", dict(max_points_per_object=k), before)
    return ctx.result("\n".join(lines), "particles")


def _build_request(ctx: ToolContext, reward_code: str, budget_interactions: int,
                   episode_length: int, action_mode: str, control_gripper: bool,
                   workspace_half_extent: float, out_dir: Optional[Path]) -> RLRequest:
    fn = load_reward(reward_code)
    snap = ctx.session.particles(ctx.rl_defaults.get("points_per_object", 32))
    value = validate_reward(fn, snap)
    d = ctx.rl_defaults
    remaining = ctx.session.budget.remaining
    budget = int(min(max(100, budget_interactions), remaining))
    req = RLRequest(
        reward_fn=fn,
        reward_source=reward_code,
        budget_interactions=budget,
        episode_length=int(np.clip(episode_length, 5, 500)),
        action_mode=action_mode,
        control_gripper=bool(control_gripper),
        workspace_half_extent=float(np.clip(workspace_half_extent, 0.02, 0.5)),
        max_step_translation=d.get("max_step_translation", 0.01),
        max_step_yaw_deg=d.get("max_step_yaw_deg", 5.0),
        points_per_object=d.get("points_per_object", 32),
        out_dir=out_dir,
        seed=ctx.session.cfg.seed + ctx.tool_calls,
        algo=d.get("algo", "sac"),
        algo_kwargs=dict(d.get("algo_kwargs", {}) or {}),
        early_stop_successes=d.get("early_stop_successes", 5),
        early_stop_window=d.get("early_stop_window", 8),
        stagnation_episodes=d.get("stagnation_episodes", 40),
        video_every=int(d.get("video_every", 2000)),
        stagnation_eps=d.get("stagnation_eps", 0.02),
        final_exec_attempts=d.get("final_exec_attempts", 3),
        progress_callback=None,
    )
    req.initial_reward = value  # type: ignore[attr-defined]
    return req


def _run_rl(ctx: ToolContext, backend_name: str, tool_name: str,
            reward_code: str, budget_interactions: int, episode_length: int,
            action_mode: str, control_gripper: bool,
            workspace_half_extent: float) -> List[Any]:
    s = ctx.session
    before = s.interactions
    if s.budget.exhausted:
        return ctx.error("Interaction budget exhausted; no more actions possible.", tool_name)
    out_dir = s.new_rl_dir()
    try:
        req = _build_request(ctx, reward_code, budget_interactions, episode_length,
                             action_mode, control_gripper, workspace_half_extent, out_dir)
    except (RewardError, ValueError) as e:
        ctx.log_call(tool_name, dict(reward_code=reward_code, error=str(e)[:500]), before)
        return ctx.error(f"Reward code rejected.\n{e}\n\n{REWARD_CONTRACT}", tool_name)
    header = (f"Reward validated on the current scene: initial value "
              f"{req.initial_reward:.4f}. Running {backend_name} with budget "  # type: ignore[attr-defined]
              f"{req.budget_interactions} interactions, episode length "
              f"{req.episode_length}, action mode {req.action_mode}"
              f"{' + gripper' if req.control_gripper else ''}, workspace half-extent "
              f"{req.workspace_half_extent:.2f} m around the current EE position.")
    s.record_event("rl_start", tool=tool_name, backend=backend_name,
                   budget=req.budget_interactions, episode_length=req.episode_length,
                   action_mode=req.action_mode, out_dir=str(out_dir),
                   initial_reward=req.initial_reward)  # type: ignore[attr-defined]
    try:
        result = ctx.backends[backend_name].run(s, req)
        text = header + "\n\n" + result.summary()
    except NotImplementedError as e:
        ctx.log_call(tool_name, dict(reward_code=reward_code, error="not_implemented"), before)
        return ctx.error(str(e), tool_name)
    except BudgetExhausted:
        text = header + "\n\nStopped: the interaction budget is exhausted."
    except Exception as e:  # pylint: disable=broad-except
        tb = traceback.format_exc(limit=5)
        s.record_event("rl_error", tool=tool_name, error=str(e), traceback=tb)
        text = header + f"\n\nRL backend failed with an internal error: {e}\n{tb}"
    ctx.log_call(tool_name, dict(reward_code=reward_code, budget=req.budget_interactions,
                                 episode_length=req.episode_length, action_mode=action_mode,
                                 control_gripper=control_gripper,
                                 workspace_half_extent=workspace_half_extent,
                                 out_dir=str(out_dir)), before)
    return ctx.result(text, tool_name)


def run_rl_on_particles(ctx: ToolContext, reward_code: str, budget_interactions: int = 20000,
                        episode_length: int = 50, action_mode: str = "xyz",
                        control_gripper: bool = False,
                        workspace_half_extent: float = 0.15) -> List[Any]:
    """Model-free RL tool."""
    return _run_rl(ctx, "model_free", "run_rl_on_particles", reward_code,
                   budget_interactions, episode_length, action_mode,
                   control_gripper, workspace_half_extent)


def run_model_based_rl_on_particles(ctx: ToolContext, reward_code: str,
                                    budget_interactions: int = 20000,
                                    episode_length: int = 50, action_mode: str = "xyz",
                                    control_gripper: bool = False,
                                    workspace_half_extent: float = 0.15) -> List[Any]:
    """Model-based RL tool (stub backend in phase 1)."""
    return _run_rl(ctx, "model_based", "run_model_based_rl_on_particles", reward_code,
                   budget_interactions, episode_length, action_mode,
                   control_gripper, workspace_half_extent)
