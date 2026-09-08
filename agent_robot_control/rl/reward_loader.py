"""Compile and validate agent-written reward functions.

Contract (also in the tool description)::

    import numpy as np
    def reward(particles: dict[str, np.ndarray],   # name -> (K, 3) world xyz
               visible: dict[str, np.ndarray],     # name -> (K,) bool
               ee_pos: np.ndarray, ee_quat: np.ndarray,  # (3,), (4,) xyzw
               gripper: float) -> float:            # 0.04 open ... 0.01 closed
        ...                                          # >= 1.0 means goal reached

Only ``numpy`` (as ``np``) and ``math`` are available, plus safe builtins.
This is a guard against accidents, not a security sandbox.
"""
from __future__ import annotations

import math
import signal
import threading
import traceback
from typing import Any, Dict

import numpy as np

from agent_robot_control.rl.backend import RewardFn

REWARD_SIGNATURE = ("def reward(particles, visible, ee_pos, ee_quat, gripper) "
                    "-> float")

_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else
    getattr(__builtins__, name)
    for name in [
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
        "isinstance", "len", "list", "max", "min", "print", "range", "round",
        "set", "sorted", "sum", "tuple", "zip", "map", "filter", "str",
        "ValueError", "KeyError", "Exception", "True", "False", "None",
    ] if (name in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, name))
}


class RewardError(ValueError):
    """The reward code could not be compiled or evaluated."""


_ALLOWED_MODULES = ("numpy", "math")


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """``__import__`` restricted to numpy and math.

    NumPy's own functions resolve some imports through the calling frame's
    builtins, so a namespace without ``__import__`` breaks even plain
    ``np.linalg.norm`` calls (KeyError: '__import__')."""
    if name.split(".")[0] in _ALLOWED_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"import of {name!r} is not allowed in reward code; "
                      "only numpy (np) and math are available.")


_SAFE_BUILTINS["__import__"] = _safe_import


def load_reward(code: str) -> RewardFn:
    """Exec ``code`` in a restricted namespace and return its ``reward``."""
    namespace: Dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "np": np,
        "numpy": np,
        "math": math,
    }
    # Allow ``import numpy as np`` / ``import math`` lines by stripping them;
    # anything else that imports is rejected.
    cleaned = []
    for line in code.splitlines():
        s = line.strip()
        if s.startswith(("import ", "from ")):
            if s.replace(" ", "") in {"importnumpyasnp", "importnumpy",
                                      "importmath", "fromnumpyimport*"}:
                continue
            raise RewardError(f"Imports are not allowed in reward code: {s!r}. "
                              "numpy (np) and math are already available.")
        cleaned.append(line)
    try:
        exec(compile("\n".join(cleaned), "<reward>", "exec"), namespace)
    except Exception as e:  # pylint: disable=broad-except
        raise RewardError("Reward code failed to compile/exec:\n"
                          + traceback.format_exc(limit=3)) from e
    fn = namespace.get("reward")
    if not callable(fn):
        raise RewardError("Reward code must define a function named 'reward' "
                          f"with signature {REWARD_SIGNATURE}.")
    return fn


class _Timeout(Exception):
    pass


def call_reward(fn: RewardFn, particles, visible, ee_pos, ee_quat, gripper,
                timeout_s: float = 2.0) -> float:
    """Call the reward with a wall-clock timeout (main thread only) and
    coerce the result to a finite float."""
    use_alarm = threading.current_thread() is threading.main_thread() and \
        hasattr(signal, "setitimer")
    if use_alarm:
        def _handler(signum, frame):
            raise _Timeout()
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        value = fn(particles, visible, ee_pos, ee_quat, gripper)
    except _Timeout as e:
        raise RewardError(f"Reward function exceeded {timeout_s}s.") from e
    except Exception as e:  # pylint: disable=broad-except
        raise RewardError("Reward function raised:\n"
                          + traceback.format_exc(limit=3)) from e
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    try:
        value = float(np.asarray(value).reshape(()))
    except Exception as e:  # pylint: disable=broad-except
        raise RewardError(f"Reward must return a scalar, got {value!r}.") from e
    if not math.isfinite(value):
        raise RewardError(f"Reward returned a non-finite value: {value!r}.")
    return value


def validate_reward(fn: RewardFn, snapshot) -> float:
    """Evaluate once on a real snapshot; raise ``RewardError`` on failure."""
    visible = {n: np.ones(len(p), dtype=bool) for n, p in snapshot.points.items()}
    return call_reward(fn, snapshot.points, visible, snapshot.ee_pos,
                       snapshot.ee_quat, snapshot.gripper)
