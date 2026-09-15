"""Atomic, identified continuation records for offline numerical inference.

These records are solver state, not posterior samples or an assessment.
JSON avoids executable deserialization; a checksum detects incomplete or
accidentally modified files. Runtime compatibility is checked by the
caller.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from predicators.code_sim_learning.inference_data import content_digest


@dataclass(frozen=True)
class SamplerCheckpoint:
    """Immutable solver state associated with one complete run signature."""
    signature: str
    state: str

    def __post_init__(self) -> None:
        if len(self.signature) != 64 or any(c not in "0123456789abcdef"
                                            for c in self.signature):
            raise ValueError("Checkpoint signature must be a SHA256 digest")
        value = json.loads(self.state)
        if not isinstance(value, dict):
            raise ValueError("Checkpoint state must be an object")
        # Reject JSON extensions for NaN/infinity, including nested values.
        json.dumps(value, allow_nan=False)

    def unpack(self) -> Dict[str, Any]:
        """Return an owned copy so callbacks cannot alter the running
        solver."""
        return dict(json.loads(self.state))

    def save(self, path: Path) -> None:
        """Atomically replace a record after flushing its complete contents.

        The parent directory must already exist. A failed write leaves
        the previous record intact; callers must serialize writers for
        the same run. No experiment is resumed automatically.
        """
        payload = json.dumps(
            {
                "schema": 1,
                "signature": self.signature,
                "state": self.state
            },
            sort_keys=True,
            allow_nan=False)
        envelope = json.dumps(
            {
                "payload": payload,
                "sha256": content_digest(payload.encode("utf-8"))
            },
            sort_keys=True,
            allow_nan=False)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w",
                                             encoding="utf-8",
                                             dir=path.parent,
                                             prefix=path.name + ".",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(envelope)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> SamplerCheckpoint:
        """Read a complete checksummed record without executing file
        content."""
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = envelope["payload"]
        if content_digest(payload.encode("utf-8")) != envelope["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        value = json.loads(payload)
        if value["schema"] != 1:
            raise ValueError("Unsupported checkpoint schema")
        return cls(value["signature"], value["state"])
