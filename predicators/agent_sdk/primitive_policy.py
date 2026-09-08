"""Guarded subprocess execution and observations for low-level policies."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict

import numpy as np
from gym.spaces import Box

from predicators.agent_sdk.primitive_policy_worker import REPLY_PREFIX
from predicators.agent_sdk.sandbox_setup import pyguard_dir, pyguard_env
from predicators.agent_sdk.tools.python_exec import _resolve_sandbox_file
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths
from predicators.structs import Action

if TYPE_CHECKING:
    from predicators.run.continual import ProtocolSession


def primitive_observation(session: ProtocolSession) -> Dict[str, Any]:
    """A detached, JSON-compatible view of the current observed frame."""
    obs = session.observe()
    objects = {
        obj.name: {
            "type": obj.type.name,
            "features": {
                name: float(obs.frame.get(obj, name))
                for name in obj.type.feature_names
            },
        }
        for obj in obs.frame
    }
    sim_state = obs.frame.simulator_state
    joints = (sim_state.get("joint_positions")
              if isinstance(sim_state, dict) else sim_state)
    return {
        "objects":
        objects,
        "joint_positions":
        (None if joints is None else [float(value) for value in joints]),
        "action_space":
        session.action_spec,
        "goal":
        obs.level.task.goal_nl or "",
        "episode_state":
        obs.state.value,
        "episode_steps":
        obs.ledger.episode_steps,
        "run_steps":
        obs.ledger.run_steps,
        "steps_remaining":
        obs.ledger.steps_remaining,
    }


def primitive_action(values: Any, space: Box) -> Action:
    """Validate before stepping, so malformed commands never cost steps."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != space.shape:
        raise ValueError(f"action must have shape {space.shape}, "
                         f"got {arr.shape}")
    if not np.all(np.isfinite(arr)) or not space.contains(arr):
        raise ValueError(
            "action must be finite and within action-space bounds")
    return Action(arr)


class PrimitivePolicy:
    """One persistent policy worker; each reply has a wall-clock deadline."""

    def __init__(self, process: asyncio.subprocess.Process,
                 timeout: float) -> None:
        self._process = process
        self._deadline = time.monotonic() + timeout

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError(
                "policy exceeded its wall-clock timeout")
        return remaining

    async def read(self) -> Dict[str, Any]:
        """Read a reply, ignoring native-library stdout diagnostics."""

        async def receive() -> Dict[str, Any]:
            assert self._process.stdout is not None
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    raise ValueError("policy worker exited without a reply")
                text = line.decode("utf-8", errors="replace")
                if text.startswith(REPLY_PREFIX):
                    result = json.loads(text[len(REPLY_PREFIX):])
                    if "error" in result:
                        raise ValueError(_scrub_host_paths(result["error"]))
                    return dict(result)

        try:
            return await asyncio.wait_for(receive(), timeout=self._remaining())
        except asyncio.TimeoutError as err:
            raise asyncio.TimeoutError(
                "policy exceeded its wall-clock timeout") from err

    async def action(self, observation: Dict[str, Any]) -> Any:
        """Ask for the next raw vector, or None to yield control."""
        assert self._process.stdin is not None
        message = json.dumps(observation, allow_nan=False) + "\n"
        self._process.stdin.write(message.encode("utf-8"))
        await asyncio.wait_for(self._process.stdin.drain(), self._remaining())
        return (await self.read())["action"]


@asynccontextmanager
async def open_primitive_policy(
        path: str, sandbox_dir: str,
        timeout: float) -> AsyncIterator[PrimitivePolicy]:
    """Load a sandbox file and always terminate its worker and children."""
    host_path, error = _resolve_sandbox_file(path, sandbox_dir)
    if error:
        raise ValueError(error)
    if not os.path.isfile(
            os.path.join(pyguard_dir(sandbox_dir), "sitecustomize.py")):
        raise ValueError("policy execution requires an initialized sandbox")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("policy timeout must be positive and finite")
    logs = os.path.join(sandbox_dir, "tool_outputs", "policies")
    os.makedirs(logs, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=logs, suffix=".log",
                                     delete=False) as stderr:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            "-m",
            "predicators.agent_sdk.primitive_policy_worker",
            host_path,
            cwd=sandbox_dir,
            env={
                **os.environ,
                **pyguard_env(sandbox_dir)
            },
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            start_new_session=True)
        try:
            worker = PrimitivePolicy(process, timeout)
            await worker.read()
            yield worker
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
