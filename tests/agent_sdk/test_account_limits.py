"""Tests for the per-account usage-limit markers."""

import os
import time

from predicators.agent_sdk.account_limits import ACCOUNT_ENV_VAR, \
    LIMIT_DIR_ENV_VAR, clear_limited, limited_until, mark_limited, \
    note_query_limited, note_query_went_through, read_limits


def test_marker_roundtrip_keeps_the_later_reset(tmp_path) -> None:
    """A marker records the reset time with private permissions, an earlier
    reset never shortens it, and an expired one reads as free."""
    assert limited_until("a", tmp_path) is None
    mark_limited("a", 1000.0, tmp_path)
    path = tmp_path / "a"
    assert path.stat().st_mode & 0o077 == 0
    assert limited_until("a", tmp_path, now=500.0) == 1000.0
    mark_limited("a", 800.0, tmp_path)
    assert limited_until("a", tmp_path, now=500.0) == 1000.0
    assert limited_until("a", tmp_path, now=1000.0) is None
    assert read_limits(["a", "b"], tmp_path, now=500.0) == {
        "a": 1000.0,
        "b": None
    }
    clear_limited("a", tmp_path)
    clear_limited("a", tmp_path)  # idempotent
    assert limited_until("a", tmp_path, now=500.0) is None


def test_harness_hooks_use_the_jobs_account(tmp_path, monkeypatch) -> None:
    """The hooks mark and clear the account named in the environment, under the
    directory the batch script exported, and do nothing for the CLI login or
    outside an account job."""
    monkeypatch.setenv(LIMIT_DIR_ENV_VAR, str(tmp_path))
    monkeypatch.delenv(ACCOUNT_ENV_VAR, raising=False)
    note_query_limited(600.0)
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv(ACCOUNT_ENV_VAR, "login")
    note_query_limited(600.0)
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv(ACCOUNT_ENV_VAR, "b")
    before = time.time()
    note_query_limited(600.0)
    until = limited_until("b")
    assert until is not None and until >= before + 600.0 - 1.0
    assert os.path.isfile(tmp_path / "b")
    note_query_went_through()
    assert limited_until("b") is None
