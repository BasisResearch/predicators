"""Model-free backend: Stable-Baselines3 SAC (default) or PPO on ParticleEnv.

Single live env, no parallel workers, no sim resets. Training stops when the
call's interaction budget is spent, the global budget is exhausted, or early
stopping fires (enough recent episodes reached reward 1). Then the learned
policy is executed deterministically from a pseudo-reset; the sim is left
wherever that ends.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from agent_robot_control.experiments.replay import VideoRecorder
from agent_robot_control.rl.backend import RLRequest, RLResult
from agent_robot_control.rl.particle_env import ParticleEnv
from agent_robot_control.sim.budget import BudgetExhausted

DEFAULT_SAC = dict(learning_rate=3e-4, buffer_size=200_000, learning_starts=500,
                   batch_size=256, tau=0.005, gamma=0.95, train_freq=1,
                   gradient_steps=1, ent_coef="auto",
                   policy_kwargs=dict(net_arch=[256, 256]))
DEFAULT_PPO = dict(learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
                   gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                   policy_kwargs=dict(net_arch=[256, 256]))


class _StopCallback(BaseCallback):
    """Budget accounting, early stopping, per-episode logging."""

    def __init__(self, env: ParticleEnv, session, request: RLRequest,
                 start_interactions: int, out_dir=None) -> None:
        super().__init__()
        self.env = env
        self.session = session
        self.req = request
        self.start = start_interactions
        self.out_dir = out_dir
        self.logged_episodes = 0
        self.early_stopped = False
        self.budget_hit = False
        self.stagnated = False
        self.videos: List[str] = []
        self._next_video_at = 0
        self._recording_episode = -1

    def _maybe_record(self, used: int) -> None:
        """Start recording at each video_every boundary; stop when that episode
        ends. One episode of video per boundary, so a long call produces a
        filmstrip of how the search changed."""
        if not self.req.video_every or self.out_dir is None:
            return
        env = self.env
        if env.recorder is not None:
            if env.episode_index != self._recording_episode:
                env.recorder.close()
                self.videos.append(str(env.recorder.path))
                env.recorder = None
            return
        if used >= self._next_video_at:
            path = self.out_dir / f"episode_{env.episode_index:04d}.mp4"
            env.recorder = VideoRecorder(path, fps=20)
            self._recording_episode = env.episode_index
            self._next_video_at = used + self.req.video_every

    def _on_step(self) -> bool:
        used = self.session.interactions - self.start
        self._maybe_record(used)
        while self.logged_episodes < len(self.env.episode_log):
            ep = self.env.episode_log[self.logged_episodes]
            self.logged_episodes += 1
            rec = {"episode": self.logged_episodes, "call_interactions": used,
                   **ep}
            self.session.record_event("rl_episode", **rec)
            if self.req.progress_callback is not None:
                self.req.progress_callback(rec)
        if used >= self.req.budget_interactions:
            self.budget_hit = True
            return False
        if self.env.dropped:
            return False
        n_stag = self.req.stagnation_episodes
        log = self.env.episode_log
        # Stagnation: only after 40% of the budget, and only if BOTH the mean
        # return and the best per-episode reward of the last window failed to
        # improve on the window before (early learning often raises the mean
        # return long before the best reward moves).
        if n_stag and len(log) >= 3 * n_stag and \
                used >= 0.4 * self.req.budget_interactions and \
                not any(e["success"] for e in log):
            recent, before = log[-n_stag:], log[-2 * n_stag:-n_stag]
            best_recent = max(e["max_reward"] for e in recent)
            best_before = max(e["max_reward"] for e in log[:-n_stag])
            mean_recent = sum(e["return"] for e in recent) / len(recent)
            mean_before = sum(e["return"] for e in before) / len(before)
            if best_recent <= best_before + self.req.stagnation_eps and \
                    mean_recent <= mean_before + self.req.stagnation_eps * self.req.episode_length:
                self.stagnated = True
                return False
        recent = self.env.episode_log[-self.req.early_stop_window:]
        if len(recent) >= self.req.early_stop_window and \
                sum(e["success"] for e in recent) >= self.req.early_stop_successes:
            self.early_stopped = True
            return False
        return True


class SB3Backend:
    """SAC or PPO behind the ``RLBackend`` protocol."""

    def run(self, session, request: RLRequest) -> RLResult:
        out_dir = Path(request.out_dir) if request.out_dir else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "reward.py").write_text(request.reward_source)
        env = ParticleEnv(session, request)
        start = session.interactions
        algo = request.algo.lower()
        kwargs: Dict[str, Any] = dict(DEFAULT_SAC if algo == "sac" else DEFAULT_PPO)
        kwargs.update(request.algo_kwargs or {})
        if algo == "sac":
            kwargs["buffer_size"] = min(kwargs["buffer_size"],
                                        max(1000, request.budget_interactions))
            model = SAC("MlpPolicy", env, seed=request.seed, verbose=0,
                        device="cpu", **kwargs)
        elif algo == "ppo":
            model = PPO("MlpPolicy", env, seed=request.seed, verbose=0,
                        device="cpu", **kwargs)
        else:
            raise ValueError(f"Unknown algo {request.algo!r}; use 'sac' or 'ppo'.")
        cb = _StopCallback(env, session, request, start, out_dir)
        budget_exhausted = False
        message = ""
        t0 = time.time()
        try:
            # total_timesteps is an upper bound; the callback stops earlier.
            model.learn(total_timesteps=int(request.budget_interactions * 1.5) + 1000,
                        callback=cb)
        except BudgetExhausted:
            budget_exhausted = True
            message = "Global interaction budget exhausted during training."
        train_time = time.time() - t0
        cb._on_step()  # flush episode log
        if env.recorder is not None:  # close a recording cut short
            env.recorder.close()
            cb.videos.append(str(env.recorder.path))
            env.recorder = None
        if cb.stagnated:
            message = (f"STOPPED EARLY: no reward improvement over the last "
                       f"{request.stagnation_episodes} episodes and no success. In a "
                       "reset-free setting this usually means the object moved out of the "
                       "policy's workspace box (or the reward has no gradient here). Look at "
                       "the scene, reposition with move_to, and reconsider the reward / box.")
        if env.dropped:
            message = (f"STOPPED EARLY: the object held at the start of this call was "
                       f"dropped at interaction {env.drop_step} (it is no longer in the "
                       "gripper). Without resets the policy cannot recover it from here. "
                       "Re-grasp it with move_to, then try again, e.g. with a smaller "
                       "workspace_half_extent or a reward that penalises large motions.")
        # Final deterministic execution from a pseudo-reset.
        final_rewards: List[float] = []
        final_success = False
        if not budget_exhausted and not env.dropped and not cb.stagnated:
            for attempt in range(request.final_exec_attempts):
                try:
                    if request.video_every and out_dir is not None:
                        env.recorder = VideoRecorder(
                            out_dir / f"final_execution_{attempt}.mp4", fps=20)
                    obs, _ = env.reset()
                    best = env.last_raw_reward
                    for _t in range(request.episode_length):
                        act, _ = model.predict(obs, deterministic=True)
                        obs, _r, term, trunc, info = env.step(act)
                        best = max(best, info["raw_reward"])
                        if term or trunc:
                            break
                    final_rewards.append(float(best))
                    if env.recorder is not None:
                        env.recorder.close()
                        cb.videos.append(str(env.recorder.path))
                        env.recorder = None
                    if best >= 1.0:
                        final_success = True
                        break
                except BudgetExhausted:
                    budget_exhausted = True
                    message = "Global interaction budget exhausted during final execution."
                    break
        policy_path = None
        if out_dir is not None:
            policy_path = str(out_dir / "policy.zip")
            model.save(policy_path)
            self._save_curve(env, out_dir)
        returns = [e["return"] for e in env.episode_log]
        result = RLResult(
            success=final_success,
            episodes=len(env.episode_log),
            interactions_used=session.interactions - start,
            successes_during_training=sum(e["success"] for e in env.episode_log),
            early_stopped=cb.early_stopped,
            budget_exhausted=budget_exhausted,
            dropped=env.dropped,
            stagnated=cb.stagnated,
            best_episode_return=float(max(returns)) if returns else float("nan"),
            last_episode_return=float(returns[-1]) if returns else float("nan"),
            final_exec_rewards=final_rewards,
            final_exec_success=final_success,
            out_dir=str(out_dir) if out_dir else None,
            policy_path=policy_path,
            message=message,
            episode_returns=returns,
            episode_max_rewards=[e["max_reward"] for e in env.episode_log],
            videos=cb.videos,
        )
        if env.recorder is not None:
            env.recorder.close()
            env.recorder = None
        if out_dir is not None:
            (out_dir / "videos.txt").write_text("\n".join(cb.videos) + "\n"
                                                if cb.videos else "")
            summary = {k: v for k, v in result.__dict__.items()}
            summary.update({"algo": algo, "algo_kwargs": {k: str(v) for k, v in kwargs.items()},
                            "train_time_s": train_time, "ik_failures": env.ik_failures,
                            "force_violations": env.force_violations,
                            "object_names": env.names})
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        session.record_event("rl_done", **{k: v for k, v in result.__dict__.items()
                                           if k not in ("episode_returns", "episode_max_rewards")})
        return result

    @staticmethod
    def _save_curve(env: ParticleEnv, out_dir: Path) -> None:
        if not env.episode_log:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(9, 3))
            x = [e["interactions"] for e in env.episode_log]
            ax[0].plot(x, [e["return"] for e in env.episode_log]); ax[0].set_title("episode return")
            ax[1].plot(x, [e["max_reward"] for e in env.episode_log]); ax[1].axhline(1.0, ls="--", c="k")
            ax[1].set_title("max raw reward in episode")
            for a in ax: a.set_xlabel("interactions")
            fig.tight_layout(); fig.savefig(out_dir / "training_curve.png", dpi=100); plt.close(fig)
        except Exception:  # pylint: disable=broad-except
            pass
