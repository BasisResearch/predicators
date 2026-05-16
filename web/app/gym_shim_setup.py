"""Builds a tiny pure-Python `gym` shim wheel.

predicators only uses `gym.spaces.Box` and (rarely) bare `import gym`,
but the full gym package has C extensions in newer versions and isn't
on Pyodide. A 50-line shim is enough to satisfy the imports envs need.

Generates `gym_shim-0.0.1-py3-none-any.whl` next to this file.
"""

from pathlib import Path
import zipfile
import textwrap

HERE = Path(__file__).parent

SPACES_PY = textwrap.dedent('''
"""Minimal gym.spaces shim — Box only, with the API surface predicators
needs (low/high arrays, shape tuple, dtype, sample, contains, seed)."""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
        if shape is None:
            low_arr = np.asarray(low, dtype=dtype)
            high_arr = np.asarray(high, dtype=dtype)
            shape = low_arr.shape if low_arr.ndim else (1,)
        else:
            shape = tuple(shape)
            low_arr = np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy()
            high_arr = np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy()
        self.low = low_arr.astype(dtype)
        self.high = high_arr.astype(dtype)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._rng = np.random.default_rng(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(self):
        return self._rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x, dtype=self.dtype)
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self) -> str:
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})"
''').strip() + "\n"

INIT_PY = '"""Minimal gym shim — predicators only needs gym.spaces.Box."""\n'

METADATA = textwrap.dedent('''
Metadata-Version: 2.1
Name: gym
Version: 0.26.2
Summary: Minimal pure-Python gym shim with only the bits predicators uses.
Author: predicators-pyodide-shim
Requires-Python: >=3.8

This is NOT the real gym package — it is a tiny shim that exposes
``gym.spaces.Box`` for use under Pyodide where the real gym
distribution can't be installed.
''').strip() + "\n"

WHEEL = textwrap.dedent('''
Wheel-Version: 1.0
Generator: predicators-pyodide-shim
Root-Is-Purelib: true
Tag: py3-none-any
''').strip() + "\n"

RECORD_LINES = [
    "gym/__init__.py",
    "gym/spaces/__init__.py",
    "gym-0.26.2.dist-info/METADATA",
    "gym-0.26.2.dist-info/WHEEL",
    "gym-0.26.2.dist-info/RECORD",
]

def _record() -> str:
    # Pyodide does not verify hashes, leave them blank.
    return "\n".join(f"{p},," for p in RECORD_LINES) + "\n"


def main() -> Path:
    out = HERE.parent / "wheels" / "gym-0.26.2-py3-none-any.whl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("gym/__init__.py", INIT_PY)
        z.writestr("gym/spaces/__init__.py", SPACES_PY)
        z.writestr("gym-0.26.2.dist-info/METADATA", METADATA)
        z.writestr("gym-0.26.2.dist-info/WHEEL", WHEEL)
        z.writestr("gym-0.26.2.dist-info/RECORD", _record())
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    main()
