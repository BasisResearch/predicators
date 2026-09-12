"""Explicit working-directory inputs for isolated offline replay workers.

Freeze optional-file absence as well as contents, and supply the
complete child environment explicitly. These inputs are one component of
runtime identity. The interpreter, imports, native libraries, assets,
command and configuration must also be captured by the caller. This is
not automatic dependency discovery, a filesystem sandbox, or a
production agent change.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Optional, Tuple

from predicators.code_sim_learning.inference_recording import ArtifactBundle, \
    SourceArtifact


@dataclass(frozen=True)
class RuntimeFile:
    """Owned file bytes, or explicit absence, at a relative worker path."""
    path: str
    content: Optional[bytes]

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if not self.path or path.is_absolute() or ".." in path.parts or \
                str(path) != self.path or self.path == "." or \
                "\\" in self.path or "\0" in self.path:
            raise ValueError("Runtime files need canonical relative paths")
        if self.content is not None and not isinstance(self.content, bytes):
            raise ValueError("Runtime file contents must be immutable bytes")

    @classmethod
    def read(cls, path: str, source: Path) -> RuntimeFile:
        """Snapshot a present regular file; absence must be declared
        explicitly.

        A typo or missing source must not silently become a claim that
        an optional dependency was absent. Symlinks require resolving
        and identifying their actual source before constructing this
        object.
        """
        if source.is_symlink() or not source.is_file():
            raise ValueError("Runtime source must be a present regular file")
        return cls(path, source.read_bytes())


@dataclass(frozen=True)
class RuntimeInputs:
    """Immutable inputs to a fresh worker directory and explicit environment.

    Use separate processes, not temporary chdir/environ mutations in
    concurrent sampler threads. The worker receives environment_dict as
    its entire environment; do not merge it with an ambient environment.
    Declared inputs are not writable output locations. Verification
    after a run catches persistent changes, not transient writes or
    external accesses. Programs that require either need a broader
    runtime contract.
    """
    files: Tuple[RuntimeFile, ...]
    environment: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        files = tuple(sorted(self.files, key=lambda f: f.path))
        names = {f.path for f in files}
        if len(names) != len(files):
            raise ValueError("Duplicate runtime file path")
        if any(
                str(parent) in names for name in names
                for parent in PurePosixPath(name).parents):
            raise ValueError("Runtime file paths cannot contain one another")
        environment = tuple(sorted(self.environment))
        if len({key for key, _ in environment}) != len(environment):
            raise ValueError("Duplicate runtime environment key")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
               or not isinstance(value, str) or "\0" in value
               for key, value in environment):
            raise ValueError("Invalid runtime environment entry")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "environment", environment)

    @property
    def environment_dict(self) -> Dict[str, str]:
        """An owned complete child environment, with no implicit
        inheritance."""
        return dict(self.environment)

    @property
    def artifacts(self) -> ArtifactBundle:
        """Persist the policy, including absent paths, with the owned bytes."""
        policy = json.dumps(
            {
                "schema":
                1,
                "kind":
                "isolated_working_directory_inputs",
                "files":
                [(f.path, "absent" if f.content is None else "present")
                 for f in self.files],
                "complete_child_environment":
                self.environment
            },
            sort_keys=True).encode("utf-8")
        return ArtifactBundle(
            (SourceArtifact("runtime-inputs.json", policy), ) + tuple(
                SourceArtifact("files/" + f.path, f.content)
                for f in self.files if f.content is not None))

    @property
    def digest(self) -> str:
        """Changed contents, absence or environment define different inputs."""
        return self.artifacts.digest

    def materialize(self, directory: Path) -> None:
        """Populate a fresh directory without reading an ambient sidecar."""
        if directory.is_symlink():
            raise ValueError("Runtime directory cannot be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError("Runtime directory must be empty")
        for source in self.files:
            if source.content is None:
                continue
            target = directory / source.path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.content)
        self.verify(directory)

    def verify(self, directory: Path) -> None:
        """Reject missing, changed, unexpected and symlinked worker inputs.

        Call both before execution and before accepting its result. This
        comparison is not a security boundary against concurrent or
        adversarial filesystem mutation.
        """
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Runtime directory must be a regular directory")
        expected = {
            f.path: f.content
            for f in self.files if f.content is not None
        }
        expected_dirs = {
            str(parent)
            for name in expected for parent in PurePosixPath(name).parents
            if str(parent) != "."
        }
        seen = set()
        for path in directory.rglob("*"):
            name = path.relative_to(directory).as_posix()
            if path.is_symlink():
                raise ValueError(f"Symlinked runtime input: {name}")
            if path.is_dir() and name in expected_dirs:
                continue
            if name not in expected or not path.is_file() or \
                    path.read_bytes() != expected[name]:
                raise ValueError(
                    f"Unexpected or changed runtime input: {name}")
            seen.add(name)
        if seen != set(expected):
            raise ValueError("Missing runtime input")
