"""Thin RPC handle the sim side holds to talk to a robot-side bridge server.

No hardware dependencies: this only sends JSON over a plain socket. The server
lives in the robot's own environment, holds the live robot connection, and
executes what it is sent.

Transport: one length-prefixed JSON request/response per RPC over TCP. The
server is persistent; the client connects per call.

Scope: open-loop execution. The caller rolls a plan out in sim, buffers the
joint-target actions, and hands them to ``execute_actions``, which splits them
into move / gripper segments and ships them in a single RPC.
"""
from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, List, Sequence

from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.structs import Action, Array


def _send_json(sock: socket.socket, obj: Any) -> None:
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_json(sock: socket.socket) -> Any:
    raw_len = _recv_exactly(sock, 4)
    (length, ) = struct.unpack(">I", raw_len)
    return json.loads(_recv_exactly(sock, length).decode("utf-8"))


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("bridge server closed the connection")
        buf.extend(chunk)
    return bytes(buf)


class BridgeServerError(RuntimeError):
    """The bridge server reported a failure executing an RPC."""


class RealRobotBridgeClient:
    """Handle for a robot-side bridge server; imports no hardware."""

    def __init__(self, host: str, port: int, timeout: float = 300.0) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout

    def _rpc(self, method: str, params: Dict[str, Any]) -> Any:
        with socket.create_connection((self._host, self._port),
                                      timeout=self._timeout) as sock:
            _send_json(sock, {"method": method, "params": params})
            resp = _recv_json(sock)
        if not resp.get("ok", False):
            raise BridgeServerError(
                f"{method} failed on the bridge server: {resp.get('error')}")
        return resp.get("result")

    def go_home(self, arm_joints: Sequence[float]) -> None:
        """Move the arm to ``arm_joints`` -- the home config the plan starts
        from -- so the plan's trajectories start where the robot actually is.

        Blocking; streamed from the robot's current pose on the server.
        """
        self._rpc("go_home", {"joints": [float(v) for v in arm_joints]})

    def execute_actions(self, actions: Sequence[Action],
                        robot: SingleArmPyBulletRobot) -> None:
        """Split buffered joint-target actions into move/gripper segments and
        execute them on the robot (blocking)."""
        segments = self._split_actions(actions, robot)
        self._rpc("execute_segments", {"segments": segments})

    @staticmethod
    def _split_actions(actions: Sequence[Action],
                       robot: SingleArmPyBulletRobot) -> List[Dict[str, Any]]:
        """Joint-target actions -> [{move, waypoints} | {gripper, command}].

        A finger value nearer ``closed_fingers`` than ``open_fingers``
        is a ``close``; consecutive same-gripper steps coalesce into one
        move of arm waypoints, with the finger joints removed.
        """
        gidx = (robot.left_finger_joint_idx, robot.right_finger_joint_idx)
        closed, opened = robot.closed_fingers, robot.open_fingers

        def grip(arr: Array) -> str:
            v = float(arr[gidx[0]])
            return "close" if abs(v - closed) < abs(v - opened) else "open"

        def arm_only(arr: Array) -> List[float]:
            return [float(v) for i, v in enumerate(arr) if i not in gidx]

        segments: List[Dict[str, Any]] = []
        cur_grip: str = ""
        moves: List[List[float]] = []
        for action in actions:
            arr = action.arr
            g = grip(arr)
            if g != cur_grip:
                if moves:
                    segments.append({"type": "move", "waypoints": moves})
                    moves = []
                segments.append({"type": "gripper", "command": g})
                cur_grip = g
            moves.append(arm_only(arr))
        if moves:
            segments.append({"type": "move", "waypoints": moves})
        return segments
