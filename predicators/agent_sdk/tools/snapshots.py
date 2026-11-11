"""Write-time versioned snapshots of agent-edited sandbox files."""
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Sim-learning tools ───────────────────────────────────────────


def format_cycle_label(cycle_idx: int) -> str:
    """Render a cycle index for snapshot filenames and version tags.

    Online cycles use the harness's 0-based "ONLINE LEARNING CYCLE i"
    numbering, zero-padded (``000``, ``001``, ...). A negative index
    denotes the offline (pre-cycle-0) learning pass and renders as
    ``offline`` so it can never be mistaken for cycle 0's online pass.
    """
    return "offline" if cycle_idx < 0 else f"{cycle_idx:03d}"


class _SnapshotTarget:  # pylint: disable=too-few-public-methods
    """One file to watch for write-time snapshots."""

    def __init__(
        self,
        live_file: str,
        versions_dir: str,
        artifact_name: str,
        cycle_index_provider: Callable[[], int],
    ) -> None:
        self.live_file = os.path.realpath(live_file)
        self.versions_dir = versions_dir
        self.artifact_name = artifact_name
        self.cycle_index_provider = cycle_index_provider


def make_write_snapshot_hook(
    targets: List[_SnapshotTarget],
    sandbox_dir: str,
) -> Callable[..., Any]:
    """Build a PostToolUse hook that snapshots target files on Write/Edit.

    The returned async callable matches the Claude Agent SDK's hook
    signature ``(hook_input, tool_use_id, hook_context) -> dict``, where
    ``hook_input`` is the CLI's JSON parsed into a dict
    (``PostToolUseHookInput``). It fires after a successful Write / Edit
    / MultiEdit and, if the tool's ``file_path`` (resolved against
    ``sandbox_dir``) matches any target's ``live_file``, writes a new
    versioned snapshot (via :func:`finalize_versioned_snapshot`).

    Dedup by content means a no-op Edit that produces identical content
    leaves no new file. Failures are swallowed — a snapshot hook
    failing should never break the agent's edit loop.
    """
    abs_sandbox = os.path.abspath(sandbox_dir)

    def _resolve(path: str) -> str:
        if os.path.isabs(path):
            return os.path.realpath(path)
        return os.path.realpath(os.path.join(abs_sandbox, path))

    target_by_path: Dict[str,
                         _SnapshotTarget] = {t.live_file: t
                                             for t in targets}

    async def _hook(hook_input: Any, _tool_use_id: Any,
                    _context: Any) -> Dict[str, Any]:
        try:
            tool_name = hook_input.get("tool_name")
            if tool_name not in {"Write", "Edit", "MultiEdit"}:
                return {}
            tool_input = hook_input.get("tool_input") or {}
            raw_path = tool_input.get("file_path")
            if not raw_path:
                return {}
            resolved = _resolve(raw_path)
            target = target_by_path.get(resolved)
            if target is None:
                return {}
            finalize_versioned_snapshot(
                target.live_file,
                target.versions_dir,
                cycle_idx=int(target.cycle_index_provider()),
                artifact_name=target.artifact_name,
            )
        except Exception:  # pylint: disable=broad-except
            # Never let a snapshot failure break the agent's edit loop.
            pass
        return {}

    return _hook


def _latest_snapshot(versions_dir: str, cycle_label: str,
                     artifact_name: str) -> Tuple[int, Optional[str]]:
    """The highest ``vers_YYY`` among the cycle's snapshots of the artifact in
    ``versions_dir``, and that snapshot's path; ``(0, None)`` if the cycle has
    none."""
    prefix = f"cycle_{cycle_label}_vers_"
    suffix = f"_{artifact_name}.py"
    highest_vers = 0
    highest_path: Optional[str] = None
    if os.path.isdir(versions_dir):
        for name in os.listdir(versions_dir):
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            vers_str = name[len(prefix):-len(suffix)]
            try:
                vers = int(vers_str)
            except ValueError:
                continue
            if vers > highest_vers:
                highest_vers = vers
                highest_path = os.path.join(versions_dir, name)
    return highest_vers, highest_path


def _write_versioned_snapshot(raw: bytes, versions_dir: str, cycle_idx: int,
                              artifact_name: str) -> str:
    """Snapshot ``raw`` as the cycle's next version unless the cycle's highest
    version holds it already; return the version's tag.

    The numbering continues from the files in ``versions_dir``, so
    every writer (the write hook, the synthesis tools' snapshotters,
    :func:`finalize_versioned_snapshot`, and a restarted run that
    re-issues a cycle) extends one sequence. A version file is created
    exclusively and never written over: when another writer takes the
    number after the scan, the snapshot takes the next one.
    """
    cycle_label = format_cycle_label(cycle_idx)
    os.makedirs(versions_dir, exist_ok=True)
    while True:
        vers, path = _latest_snapshot(versions_dir, cycle_label, artifact_name)
        if path is not None:
            with open(path, "rb") as f:
                if f.read() == raw:
                    return f"cycle_{cycle_label}_vers_{vers:03d}"
        tag = f"cycle_{cycle_label}_vers_{vers + 1:03d}"
        try:
            with open(os.path.join(versions_dir, f"{tag}_{artifact_name}.py"),
                      "xb") as f:
                f.write(raw)
        except FileExistsError:
            continue
        return tag


def finalize_versioned_snapshot(
    live_file: str,
    versions_dir: str,
    cycle_idx: int,
    artifact_name: str,
) -> Optional[str]:
    """Take a final ``cycle_XXX_vers_(YYY+1)`` snapshot if needed.

    Called from the approach after the agent session ends so that any
    post-evaluation edits to ``live_file`` (which would otherwise be
    lost — the synthesis tools only snapshot on eval calls) are
    captured. If the live file matches the highest existing
    ``cycle_XXX_vers_YYY_<artifact_name>.py`` in ``versions_dir`` (this
    cycle), the existing tag is returned and no new file is written.

    Args:
        live_file: Host path to the file (e.g. simulator.py).
        versions_dir: Directory containing the per-call snapshots.
        cycle_idx: Current cycle (0-based, matching the harness's
            "ONLINE LEARNING CYCLE i"; negative = the offline pass,
            rendered as ``offline``) — used to find the highest
            existing ``vers_YYY`` for this cycle and to name the new
            snapshot.
        artifact_name: Stem used in the filename, e.g. ``"simulator"``
            or ``"predicates"``.

    Returns the final version tag (``cycle_XXX_vers_YYY``) or ``None``
    if ``live_file`` does not exist.
    """
    if not os.path.isfile(live_file):
        return None
    with open(live_file, "rb") as f:
        live_raw = f.read()
    return _write_versioned_snapshot(live_raw, versions_dir, cycle_idx,
                                     artifact_name)


def restored_version(live_file: str, versions_dir: str, cycle_idx: int,
                     artifact_name: str,
                     checkpoint_tag: Optional[str]) -> Optional[str]:
    """The version tag of an artifact restored from a checkpoint.

    The checkpoint's tag when its snapshot holds the restored file;
    otherwise (a checkpoint taken before the file's last edit, or before
    it existed) the tag :func:`finalize_versioned_snapshot` gives it.
    ``None`` if ``live_file`` does not exist.
    """
    if checkpoint_tag is not None and os.path.isfile(live_file):
        snapshot = os.path.join(versions_dir,
                                f"{checkpoint_tag}_{artifact_name}.py")
        if os.path.isfile(snapshot):
            with open(snapshot, "rb") as f:
                held = f.read()
            with open(live_file, "rb") as f:
                if f.read() == held:
                    return checkpoint_tag
    return finalize_versioned_snapshot(live_file, versions_dir, cycle_idx,
                                       artifact_name)


class _ArtifactSnapshotter:
    """Per-call versioned snapshotting for one artifact file.

    Used by the synthesis-tools factories to snapshot the file each tool
    call loads and tag the load with ``cycle_XXX_vers_YYY``. ``XXX`` is
    read from ``cycle_index_provider`` at each call so live cycle bumps
    are reflected in subsequent tags. ``YYY`` continues from the cycle's
    highest version on disk (see :func:`_write_versioned_snapshot`), so
    a new snapshotter, the write hook and a restarted run's snapshotter
    all extend one sequence; unchanged content keeps the tag of the
    snapshot that holds it.
    """

    def __init__(
        self,
        live_file: str,
        versions_dir: str,
        artifact_name: str,
        cycle_index_provider: Optional[Callable[[], int]],
        missing_file_hint: str = "",
    ) -> None:
        self._live_file = live_file
        self._versions_dir = versions_dir
        self._artifact_name = artifact_name
        self._cycle_index_provider = cycle_index_provider
        self._missing_file_hint = missing_file_hint

    def current_cycle(self) -> int:
        """Return the active learning-cycle index, or 0 if unknown."""
        if self._cycle_index_provider is None:
            return 0
        try:
            return int(self._cycle_index_provider())
        except Exception:  # pylint: disable=broad-except
            return 0

    def snapshot(
        self,
        path: Optional[str] = None,
    ) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
        """Read the live file and write a versioned snapshot on change.

        Returns ``(raw_bytes, version_tag, error_msg)``. On a missing
        file, ``raw_bytes`` and ``version_tag`` are ``None`` and
        ``error_msg`` carries a user-facing message (suffixed with
        ``missing_file_hint`` when configured).

        ``path`` may override the configured ``live_file`` per call —
        the snapshotter still writes into the configured
        ``versions_dir`` under ``artifact_name``, so both files extend
        one version sequence and dedup spans both.
        """
        target = path or self._live_file
        if not os.path.isfile(target):
            msg = (f"{self._artifact_name.capitalize()} file not found: "
                   f"{target}.")
            if self._missing_file_hint:
                msg = f"{msg} {self._missing_file_hint}"
            return None, None, msg
        with open(target, "rb") as f:
            raw = f.read()
        tag = _write_versioned_snapshot(raw, self._versions_dir,
                                        self.current_cycle(),
                                        self._artifact_name)
        return raw, tag, None
