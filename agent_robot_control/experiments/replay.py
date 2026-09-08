"""Deterministically replay a run's tool calls, optionally recording a video.

    uv run python -m agent_robot_control.experiments.replay <run_dir> \
        [--video out.mp4] [--fps 30] [--stop-at-success] [--camera 640x360]

PyBullet is deterministic for a fixed action sequence, so a run whose first
success happens before its first ``run_rl_on_particles`` call is reproduced
exactly from ``events.jsonl`` (replayed interaction counts match the log).
Runs that succeed during an RL call cannot be replayed this way; the replay
stops at the first RL call and says so.

Every env step is rendered as one video frame with a caption strip showing
the tool being executed, the interaction counter and the goal state.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List, Optional  # noqa: E402

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

CAPTION_H = 34


class VideoRecorder:
    """Writes captioned frames of the live sim to an mp4."""

    def __init__(self, path: Path, fps: int = 30, every: int = 1) -> None:
        import imageio.v2 as imageio
        self.path = path
        self.every = max(1, every)
        self.frames = 0
        self.calls = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(path), fps=fps,
                                          macro_block_size=1, quality=8)

    def add(self, rgb: np.ndarray, caption: str) -> None:
        """Append one frame with ``caption`` drawn under the image."""
        from PIL import Image, ImageDraw
        self.calls += 1
        if (self.calls - 1) % self.every:
            return
        h, w = rgb.shape[:2]
        canvas = Image.new("RGB", (w, h + CAPTION_H), (16, 16, 16))
        canvas.paste(Image.fromarray(rgb), (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, h + 6), caption[:120], fill=(235, 235, 235))
        self._writer.append_data(np.asarray(canvas))
        self.frames += 1

    def close(self) -> None:
        """Finish the file."""
        self._writer.close()


def replay(run_dir: Path,
           video: Optional[Path] = None,
           fps: int = 30,
           every: int = 1,
           camera: Optional[str] = None,
           stop_at_success: bool = False,
           max_interactions: Optional[int] = None,
           tail_frames: int = 45,
           verbose: bool = True) -> dict:
    """Replay ``run_dir`` and return a summary dict."""
    from agent_robot_control.mcp_server.server import build_session
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.run_dir = None
    cfg.server.log_transitions = False
    if camera:
        w, h = camera.lower().split("x")
        cfg.server.camera_width, cfg.server.camera_height = int(w), int(h)
    session = build_session(cfg)
    env, ctl = session.env, session.controller
    label = "/".join(run_dir.parts[-4:])
    goal_text = ", ".join(sorted(str(a) for a in session.task.goal))
    ev = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    calls = [e for e in ev if e["kind"] == "tool_call"]
    logged_goal = next((e["interactions"] for e in ev if e["kind"] == "goal_reached"), None)

    rec = VideoRecorder(video, fps=fps, every=every) if video else None
    state = {"tool": "start", "args": ""}

    def caption() -> str:
        status = "SUCCESS" if session.goal_reached_now else "running"
        return (f"{label} | {state['tool']} {state['args']} | "
                f"step {session.interactions} | {status} | goal: {goal_text}")

    # Record every env step by wrapping the session's step function.
    inner_step = session.step

    def recording_step(action):
        inner_step(action)
        if rec is not None:
            rec.add(session.render(), caption())

    session.step = recording_step  # type: ignore[assignment]
    ctl.step_fn = recording_step

    if rec is not None:
        state["tool"] = "initial state"
        for _ in range(15):
            rec.add(session.render(), caption())

    stopped = None
    mismatch = 0
    for i, e in enumerate(calls):
        tool, a = e["tool"], e.get("args", {})
        if tool.startswith("run_"):
            stopped = (f"call {i} is {tool}: RL is not replayable, stopping "
                       "(the original run's success may lie inside it)")
            break
        if tool == "pixels_to_particles":
            continue  # costs no interactions and does not move anything
        state["tool"] = tool
        if tool == "move_to":
            state["args"] = (f"({a['x']:.3f}, {a['y']:.3f}, {a['z']:.3f}) "
                             f"yaw={a.get('yaw', 0):.0f} grip={a.get('gripper', 'keep')}")
            q = ctl.quat_from_rpy_deg(a.get("roll", 0), a.get("pitch", 0), a.get("yaw", 0))
            ctl.move_to((a["x"], a["y"], a["z"]), q,
                        gripper=a.get("gripper", "keep"),
                        max_steps=int(a.get("max_steps", 200)))
        elif tool == "wait":
            state["args"] = f"{a['steps']} steps"
            for _ in range(int(a["steps"])):
                recording_step(ctl.hold_action())
        else:
            continue
        if e.get("interactions_after") != session.interactions:
            mismatch += 1
        if verbose:
            print(f"call {i:3d} {tool:20s} logged={e.get('interactions_after'):5d} "
                  f"replay={session.interactions:5d} goal={session.goal_reached_now}")
        if stop_at_success and session.goal_reached_now:
            stopped = f"first success at interaction {session.interactions}"
            break
        if max_interactions and session.interactions >= max_interactions:
            stopped = f"reached --max-interactions {max_interactions}"
            break

    if rec is not None:
        state["tool"] = "SUCCESS" if session.goal_reached_now else "end of replay"
        state["args"] = ""
        for _ in range(tail_frames):
            rec.add(session.render(), caption())
        rec.close()
    summary = {
        "run": str(run_dir),
        "logged_first_success": logged_goal,
        "replay_first_success": session.goal_first_reached_at,
        "replay_interactions": session.interactions,
        "count_mismatches": mismatch,
        "stopped": stopped,
        "video": str(video) if video else None,
        "frames": rec.frames if rec else 0,
    }
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every", type=int, default=1, help="record every Nth step")
    ap.add_argument("--camera", default=None, help="render size, e.g. 640x360")
    ap.add_argument("--stop-at-success", action="store_true")
    ap.add_argument("--max-interactions", type=int, default=None)
    args = ap.parse_args()
    replay(args.run_dir, video=args.video, fps=args.fps, every=args.every,
           camera=args.camera, stop_at_success=args.stop_at_success,
           max_interactions=args.max_interactions)


if __name__ == "__main__":
    main()
