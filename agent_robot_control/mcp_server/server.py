"""Standalone stdio MCP server exposing the robot tools.

Launch (the harness does this from its MCP config)::

    uv run python -m agent_robot_control.mcp_server.server --run-dir <dir>

``<dir>/config.yaml`` is the resolved experiment config (written by the
experiment runner); ``--config`` can point at any yaml with the same layout.
The server owns the one live simulation for the life of the harness session.
Tool bodies run in a worker thread (serialised by a lock) so the event loop
keeps answering pings during long RL calls.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import asyncio  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

from mcp.server.mcpserver import MCPServer  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from agent_robot_control.mcp_server import tools as T  # noqa: E402
from agent_robot_control.rl.model_based_backend import ModelBasedBackend  # noqa: E402
from agent_robot_control.rl.sb3_backend import SB3Backend  # noqa: E402
from agent_robot_control.sim.session import SessionConfig, SimSession  # noqa: E402

ALL_TOOLS = ["move_to", "wait", "pixels_to_particles", "run_rl_on_particles",
             "run_model_based_rl_on_particles"]

TOOL_DESCRIPTIONS = {
    "move_to": (
        "Move the robot end-effector (gripper) to a target pose and optionally "
        "open/close the gripper. Blocking; returns when the target is reached, "
        "the arm stalls (blocked by contact), inverse kinematics fails, or "
        "max_steps env steps are used. Arguments: x, y, z target position in "
        "metres (world frame); roll, pitch, yaw in degrees applied on top of "
        "the default gripper-down orientation (yaw rotates about the vertical "
        "axis; leave 0 unless needed); gripper 'open' | 'close' | 'keep'; "
        "max_steps (default 200; each step moves about 2 cm). The parallel-jaw gripper opens to about 8 cm between fingertips and closes to about 2 cm; objects up to ~7 cm wide can be grasped. Every env step "
        "counts against the interaction budget. Returns the outcome, the final "
        "EE pose, gripper opening, budget status, and the current camera image."),
    "wait": (
        "Hold the arm still for `steps` environment steps so that time passes "
        "(e.g. to let a conveyor belt bring an item into position). Each step "
        "counts against the interaction budget, like every other action. "
        "Returns the camera image afterwards. Use this instead of small "
        "move_to calls when you only need to wait."),
    "pixels_to_particles": (
        "Render the camera and convert the depth image into per-object 3D "
        "point clouds ('particles') using the object segmentation. Writes a "
        ".npz file (arrays points_<name> (K,3) world xyz in metres, "
        "colors_<name>, ee_pos, ee_quat_xyzw, gripper) and a .json summary "
        "(per-object centroid and bounding box) into the working directory, "
        "and returns the summary. Costs no interactions. Use it to locate "
        "objects precisely before moving or to design a reward."),
    "run_rl_on_particles": (
        "Run model-free reinforcement learning (SAC or PPO) in the live "
        "environment to maximise a reward you write over the particles. Use "
        "this for fine-grained motor control that move_to cannot do reliably "
        "(precise alignment, insertion, nudging). The policy acts in small "
        "end-effector deltas (about 1 cm per step) inside a box of half-extent "
        "workspace_half_extent around the CURRENT end-effector position, so "
        "first move_to a good starting pose. There are NO environment resets: "
        "between episodes only the arm returns to that starting pose; objects "
        "stay where they are. Arguments: reward_code (Python source defining "
        "def reward(particles, visible, ee_pos, ee_quat, gripper) -> float, "
        "where >= 1.0 means success; see the tool result on error for the full "
        "contract), budget_interactions (env steps this call may use), "
        "episode_length (steps per episode, default 50), action_mode 'xyz' or "
        "'xyz_yaw', control_gripper (let the policy open/close the gripper), "
        "workspace_half_extent (metres, default 0.15). Training stops early "
        "once recent episodes reliably reach reward 1. Afterwards the learned "
        "policy is executed and the environment is left in the resulting state. "
        "This call can take a long time. Returns a training summary, the "
        "interactions spent, and the camera image after execution."),
    "run_model_based_rl_on_particles": (
        "Like run_rl_on_particles but learns a particle-based world model and "
        "plans with it (model-based RL). Same arguments and reward contract. "
        "NOTE: this backend may report that it is not implemented yet; in that "
        "case use run_rl_on_particles."),
}


def build_session(cfg) -> SimSession:
    """Create the SimSession from a resolved config (OmegaConf)."""
    env_cfg = cfg.get("env", {})
    srv = cfg.get("server", {})
    overrides = dict(env_cfg.get("class_overrides", {}) or {})
    tier = env_cfg.get("clearance_tier")
    if tier and env_cfg.get("name") == "pybullet_plug_outlet":
        from predicators.envs.pybullet_plug_outlet import PyBulletPlugOutletEnv
        overrides["clearance"] = PyBulletPlugOutletEnv.clearance_tiers[tier]
    scfg = SessionConfig(
        env_name=env_cfg.get("name", "pybullet_donut"),
        task_idx=int(env_cfg.get("task_idx", 0)),
        seed=int(cfg.get("seed", 0)),
        interaction_cap=int(srv.get("interaction_cap", 100_000)),
        camera_width=int(srv.get("camera_width", 335)),
        camera_height=int(srv.get("camera_height", 180)),
        particles_per_object=int(srv.get("particles_per_object", 64)),
        log_transitions=bool(srv.get("log_transitions", True)),
        transition_particles_per_object=int(srv.get("transition_particles_per_object", 32)),
        use_urdf_torque_limits=bool(srv.get("use_urdf_torque_limits", True)),
        torque_limit_scale=float(srv.get("torque_limit_scale", 3.0)),
        max_contact_force=float(srv.get("max_contact_force", 80.0)),
        env_overrides=overrides,
        run_dir=str(cfg.get("run_dir")) if cfg.get("run_dir") else None,
    )
    return SimSession(scfg)


def build_context(cfg, session: Optional[SimSession] = None) -> T.ToolContext:
    """Session + backends + RL defaults."""
    session = session or build_session(cfg)
    rl = OmegaConf.to_container(cfg.get("rl", {}), resolve=True) if cfg.get("rl") is not None else {}
    backends = {"model_free": SB3Backend(), "model_based": ModelBasedBackend()}
    return T.ToolContext(session, backends, rl or {})


def make_server(ctx: T.ToolContext, tool_names: List[str]) -> MCPServer:
    """Register the selected tools on an MCPServer."""
    server = MCPServer(name="robot", version="0.1.0",
                       instructions=("Tools for controlling a simulated robot arm. "
                                     "Every env step counts against a fixed interaction budget."))
    loop_lock = ctx.lock

    def run_locked(fn, *args, **kwargs):
        with loop_lock:
            return fn(ctx, *args, **kwargs)

    if "move_to" in tool_names:
        @server.tool(name="move_to", description=TOOL_DESCRIPTIONS["move_to"])
        async def move_to(x: float, y: float, z: float, roll: float = 0.0,
                          pitch: float = 0.0, yaw: float = 0.0,
                          gripper: str = "keep", max_steps: int = 200):
            return await asyncio.to_thread(run_locked, T.move_to, x, y, z, roll,
                                           pitch, yaw, gripper, max_steps)

    if "wait" in tool_names:
        @server.tool(name="wait", description=TOOL_DESCRIPTIONS["wait"])
        async def wait(steps: int = 50):
            return await asyncio.to_thread(run_locked, T.wait, steps)

    if "pixels_to_particles" in tool_names:
        @server.tool(name="pixels_to_particles",
                     description=TOOL_DESCRIPTIONS["pixels_to_particles"])
        async def pixels_to_particles(max_points_per_object: int = 64):
            return await asyncio.to_thread(run_locked, T.pixels_to_particles,
                                           max_points_per_object)

    if "run_rl_on_particles" in tool_names:
        @server.tool(name="run_rl_on_particles",
                     description=TOOL_DESCRIPTIONS["run_rl_on_particles"])
        async def run_rl_on_particles(reward_code: str, budget_interactions: int = 20000,
                                      episode_length: int = 50, action_mode: str = "xyz",
                                      control_gripper: bool = False,
                                      workspace_half_extent: float = 0.15):
            return await asyncio.to_thread(run_locked, T.run_rl_on_particles,
                                           reward_code, budget_interactions,
                                           episode_length, action_mode,
                                           control_gripper, workspace_half_extent)

    if "run_model_based_rl_on_particles" in tool_names:
        @server.tool(name="run_model_based_rl_on_particles",
                     description=TOOL_DESCRIPTIONS["run_model_based_rl_on_particles"])
        async def run_model_based_rl_on_particles(reward_code: str,
                                                  budget_interactions: int = 20000,
                                                  episode_length: int = 50,
                                                  action_mode: str = "xyz",
                                                  control_gripper: bool = False,
                                                  workspace_half_extent: float = 0.15):
            return await asyncio.to_thread(run_locked, T.run_model_based_rl_on_particles,
                                           reward_code, budget_interactions,
                                           episode_length, action_mode,
                                           control_gripper, workspace_half_extent)
    return server


def load_config(run_dir: Optional[str], config: Optional[str], overrides: List[str]):
    """Resolved OmegaConf config from run_dir/config.yaml or --config."""
    path = Path(config) if config else (Path(run_dir) / "config.yaml" if run_dir else None)
    cfg = OmegaConf.load(path) if path and path.exists() else OmegaConf.create({})
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    if run_dir and not cfg.get("run_dir"):
        cfg.run_dir = str(run_dir)
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. env.name=pybullet_donut")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.run_dir, args.config, args.overrides)
    tool_names = list(cfg.get("condition", {}).get("tools", ALL_TOOLS[:4]))
    ctx = build_context(cfg)
    server = make_server(ctx, tool_names)
    logging.getLogger("robot-mcp").info("serving tools %s for env %s", tool_names,
                                        ctx.session.cfg.env_name)
    try:
        server.run(transport="stdio")
    finally:
        ctx.session.close()


if __name__ == "__main__":
    main()
