"""Per-account usage-limit markers shared between the harness and the launcher.

When a sandbox query is refused with a usage-limit banner, the harness
records the banner's stated reset time in a marker file named after the
account the job runs on (``PREDICATORS_CLAUDE_ACCOUNT``, exported by the
Engaging batch script; see ``scripts/engaging/claude_accounts.py``). A
task that starts later reads the markers and keeps off any account
whose reset is still ahead, which is what the launcher can do for
"most budget left" when an account's usage windows cannot be read
directly (a token without the profile scope gets a 403 from the usage
endpoint). A query that goes through clears the marker again.

Markers live in ``PREDICATORS_CLAUDE_LIMIT_DIR`` (default
``~/.claude-tokens/.limited``), one file per account holding the reset
time as seconds since the epoch, written with mode 600.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

ACCOUNT_ENV_VAR = "PREDICATORS_CLAUDE_ACCOUNT"
LIMIT_DIR_ENV_VAR = "PREDICATORS_CLAUDE_LIMIT_DIR"
LOGIN_ACCOUNT = "login"
DEFAULT_LIMIT_DIR = Path.home() / ".claude-tokens" / ".limited"


def limit_dir(default: Optional[Path] = None) -> Path:
    """The marker directory: the environment override, else ``default``, else
    the home-directory default."""
    override = os.environ.get(LIMIT_DIR_ENV_VAR)
    if override:
        return Path(override)
    return default if default is not None else DEFAULT_LIMIT_DIR


def marker_path(account: str, directory: Optional[Path] = None) -> Path:
    """Where ``account``'s marker lives."""
    return limit_dir(directory) / account


def mark_limited(account: str,
                 reset_at: float,
                 directory: Optional[Path] = None) -> None:
    """Record that ``account`` is limited until ``reset_at`` (epoch seconds); a
    marker that already names a later reset is kept."""
    path = marker_path(account, directory)
    try:
        current = _read_marker(path)
        if current is not None and current >= reset_at:
            return
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"{reset_at:.0f}\n")
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[claude-account] limit marker not written: {exc}",
              file=sys.stderr)


def clear_limited(account: str, directory: Optional[Path] = None) -> None:
    """Forget ``account``'s marker (a query went through)."""
    try:
        marker_path(account, directory).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[claude-account] limit marker not cleared: {exc}",
              file=sys.stderr)


def _read_marker(path: Path) -> Optional[float]:
    """The reset time a marker file holds, or None when it is missing or
    unreadable."""
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def limited_until(account: str,
                  directory: Optional[Path] = None,
                  now: Optional[float] = None) -> Optional[float]:
    """The reset time recorded for ``account`` when it is still ahead, else
    None (no marker, an expired one, or an unreadable one)."""
    reset_at = _read_marker(marker_path(account, directory))
    if reset_at is None or reset_at <= (time.time() if now is None else now):
        return None
    return reset_at


def read_limits(accounts: Sequence[str],
                directory: Optional[Path] = None,
                now: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Every account's live reset time, or None when it is not limited."""
    return {name: limited_until(name, directory, now) for name in accounts}


def note_query_limited(stated_wait: float,
                       directory: Optional[Path] = None) -> None:
    """Harness hook: the job's account (from the environment) is limited for
    another ``stated_wait`` seconds.

    No-op outside an account job.
    """
    account = os.environ.get(ACCOUNT_ENV_VAR, "")
    if not account or account == LOGIN_ACCOUNT:
        return
    mark_limited(account, time.time() + stated_wait, directory)


def note_query_went_through(directory: Optional[Path] = None) -> None:
    """Harness hook: a query on the job's account succeeded, so the account is
    not limited any more."""
    account = os.environ.get(ACCOUNT_ENV_VAR, "")
    if not account or account == LOGIN_ACCOUNT:
        return
    clear_limited(account, directory)
