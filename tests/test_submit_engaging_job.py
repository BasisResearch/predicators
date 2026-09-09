"""Tests for the Engaging batch-script builder.

The self-requeue trap is what lets a requeue-enabled job heal a wall-
time TIMEOUT by itself (requeue + --auto_resume), instead of relying on
an external watcher process to notice the TIMEOUT and resubmit.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

import pytest

from scripts.engaging import claude_accounts
from scripts.engaging.claude_accounts import ACCOUNTS_ENV_VAR, LOGIN_ACCOUNT, \
    POLICY_ENV_VAR, POLICY_ROUND_ROBIN, POLICY_USAGE, USAGE_FILE_ENV_VAR, \
    AccountUsage, account_block, assigned_account, describe_assignment
from scripts.engaging.claude_accounts import main as accounts_main
from scripts.engaging.claude_accounts import pick_account, read_usages, \
    resolve_accounts, same_account_warning
from scripts.engaging.submit_engaging_job import _build_batch_script


def _bash_syntax_ok(script: str, tmp_path) -> Tuple[bool, str]:
    path = tmp_path / "job.sh"
    path.write_text(script, encoding="utf-8")
    result = subprocess.run(["bash", "-n", str(path)],
                            capture_output=True,
                            check=False)
    return result.returncode == 0, result.stderr.decode("utf-8")


def test_plain_script_without_requeue(tmp_path) -> None:
    """Without requeue the script is the simple foreground command."""
    script = _build_batch_script("main.py",
                                 "--env pybullet_bridge",
                                 requeue=False)
    assert script.rstrip().endswith("--seed $SLURM_ARRAY_TASK_ID")
    assert "scontrol requeue" not in script
    assert "trap" not in script
    ok, err = _bash_syntax_ok(script, tmp_path)
    assert ok, err


def test_requeue_script_installs_self_requeue_trap(tmp_path) -> None:
    """With requeue the script traps USR1, requeues its own array task, caps
    restarts via SLURM_RESTART_COUNT, and backgrounds python so the trap can
    fire while it runs."""
    script = _build_batch_script("main.py",
                                 "--env pybullet_bridge",
                                 requeue=True)
    assert "trap _requeue_on_timeout USR1" in script
    assert "scontrol requeue" in script
    assert "SLURM_RESTART_COUNT" in script
    # python is backgrounded and reaped, so the trap can interrupt wait.
    assert "--seed $SLURM_ARRAY_TASK_ID &" in script
    assert script.count('wait "$_PY_PID"') == 2
    # Array tasks requeue by <array_job>_<task>; plain jobs by job id.
    assert "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" in script
    ok, err = _bash_syntax_ok(script, tmp_path)
    assert ok, err


# -- Claude accounts (scripts/engaging/claude_accounts.py) -----------------


def _write_token(token_dir: Path,
                 name: str,
                 token: str,
                 mode: int = 0o600) -> Path:
    token_dir.mkdir(exist_ok=True)
    path = token_dir / name
    path.write_text(token + "\n", encoding="utf-8")
    path.chmod(mode)
    return path


def _run_account_block(block: str, task_id: int,
                       **extra_env: str) -> subprocess.CompletedProcess:
    """Run the bash block as an array task and print what it exported."""
    script = (block + '\necho "account=$PREDICATORS_CLAUDE_ACCOUNT '
              'token=${CLAUDE_CODE_OAUTH_TOKEN:-unset}"\n')
    env = {
        "PATH": os.environ["PATH"],
        "SLURM_ARRAY_TASK_ID": str(task_id),
        # The round-robin policy keeps these tests off the usage endpoint;
        # the usage-policy test below overrides it with a usage file.
        POLICY_ENV_VAR: POLICY_ROUND_ROBIN,
        **extra_env
    }
    return subprocess.run(["bash", "-c", script],
                          env=env,
                          capture_output=True,
                          text=True,
                          check=False)


def test_account_block_round_robins_seeds_over_accounts(tmp_path) -> None:
    """Array task (seed) k of the experiment at offset i runs on account (i +
    k) mod n, with that account's token read from its file at run time;
    python's assigned_account mirrors the bash."""
    _write_token(tmp_path, "a", "sk-ant-oat01-AAA")
    _write_token(tmp_path, "b", "sk-ant-oat01-BBB")
    block = account_block(["a", "b"], offset=1, token_dir=tmp_path)
    assert "sk-ant-oat01" not in block  # tokens never enter the script
    expected = {0: "b", 1: "a", 2: "b"}
    for seed, name in expected.items():
        result = _run_account_block(block, seed)
        assert result.returncode == 0, result.stderr
        token = "sk-ant-oat01-" + name.upper() * 3
        assert f"account={name} token={token}" in result.stdout
        assert f"[claude-account] {name}" in result.stdout
        assert assigned_account(["a", "b"], 1, seed) == name


def test_login_account_unsets_an_inherited_token(tmp_path) -> None:
    """The reserved login account runs on the CLI's stored login even when the
    submitting shell exported a token."""
    block = account_block([LOGIN_ACCOUNT], offset=0, token_dir=tmp_path)
    result = _run_account_block(block, 3, CLAUDE_CODE_OAUTH_TOKEN="stale")
    assert result.returncode == 0, result.stderr
    assert "account=login token=unset" in result.stdout


def test_missing_or_empty_token_file_fails_the_job(tmp_path) -> None:
    """A task whose token file is missing or empty ends with an error rather
    than silently falling back to the stored login."""
    block = account_block(["a"], offset=0, token_dir=tmp_path)
    result = _run_account_block(block, 0)
    assert result.returncode == 1
    assert "no usable token" in result.stderr
    _write_token(tmp_path, "a", "   ")
    result = _run_account_block(block, 0)
    assert result.returncode == 1
    assert "no usable token" in result.stderr


def test_resolve_accounts_precedence(tmp_path, monkeypatch) -> None:
    """Explicit list, then the environment variable, then login only."""
    _write_token(tmp_path, "a", "tok-a")
    _write_token(tmp_path, "b", "tok-b")
    monkeypatch.delenv(ACCOUNTS_ENV_VAR, raising=False)
    assert resolve_accounts(None, tmp_path) == [LOGIN_ACCOUNT]
    assert resolve_accounts("", tmp_path) == [LOGIN_ACCOUNT]
    monkeypatch.setenv(ACCOUNTS_ENV_VAR, "b, login")
    assert resolve_accounts(None, tmp_path) == ["b", LOGIN_ACCOUNT]
    assert resolve_accounts("a,b", tmp_path) == ["a", "b"]


def test_resolve_accounts_rejects_unusable_token_files(tmp_path) -> None:
    """A bad account fails the launch, naming the problem."""
    with pytest.raises(ValueError, match="no token file"):
        resolve_accounts("missing", tmp_path)
    _write_token(tmp_path, "loose", "tok", mode=0o644)
    with pytest.raises(ValueError, match="readable by others"):
        resolve_accounts("loose", tmp_path)
    _write_token(tmp_path, "empty", "")
    with pytest.raises(ValueError, match="is empty"):
        resolve_accounts("empty", tmp_path)
    _write_token(tmp_path, "a", "tok-a")
    with pytest.raises(ValueError, match="listed twice"):
        resolve_accounts("a,a", tmp_path)
    with pytest.raises(ValueError, match="not a plain file name"):
        resolve_accounts("../a", tmp_path)


def test_describe_assignment_names_each_accounts_seeds() -> None:
    """The launch log line groups the seeds by account."""
    assert describe_assignment(["a", "b"], 0, 0, 3) == \
        "a <- seeds 0,2; b <- seeds 1"
    assert describe_assignment(["a", "b"], 1, 0, 3) == \
        "a <- seeds 1; b <- seeds 0,2"
    assert describe_assignment([LOGIN_ACCOUNT], 4, 456, 2) == \
        "login <- seeds 456,457"


def test_batch_script_carries_the_account_block(tmp_path) -> None:
    """The generated script selects the account before python starts, for both
    the plain and the requeue layouts, and stays valid bash."""
    _write_token(tmp_path, "a", "tok-a")
    _write_token(tmp_path, "b", "tok-b")
    for requeue in (False, True):
        script = _build_batch_script("main.py",
                                     "--env pybullet_bridge",
                                     requeue=requeue,
                                     accounts=["a", "b"],
                                     account_offset=2,
                                     token_dir=tmp_path)
        assert "_ACCOUNTS=(a b)" in script
        assert "(2 + ${SLURM_ARRAY_TASK_ID:-0}) % 2" in script
        assert script.index("_ACCOUNTS=") < script.index("python predicators")
        ok, err = _bash_syntax_ok(script, tmp_path)
        assert ok, err
    # The default is the stored login, with any inherited token dropped.
    script = _build_batch_script("main.py", "--env x", requeue=False)
    assert "_ACCOUNTS=(login)" in script
    assert "unset CLAUDE_CODE_OAUTH_TOKEN" in script


# -- Usage-based account pick ----------------------------------------------


def _usage(session: float, weekly: float) -> AccountUsage:
    return AccountUsage(session=session, weekly=weekly)


def test_pick_account_prefers_the_most_leftover_budget() -> None:
    """The account whose tightest window has the most left wins when it leads
    the round-robin choice by at least the margin."""
    usages = {"a": _usage(session=90.0, weekly=40.0), "b": _usage(30.0, 60.0)}
    # Round-robin says a (offset 0, task 0), but b has 40% left vs a's 10%.
    name, reason = pick_account(["a", "b"], 0, 0, usages)
    assert name == "b"
    assert "most budget left" in reason
    # Round-robin already says b: it stands.
    assert pick_account(["a", "b"], 0, 1, usages)[0] == "b"


def test_pick_account_keeps_round_robin_within_margin_or_without_usage(
) -> None:
    """A lead under the margin keeps the round-robin spread; unreadable usage
    means round-robin; a single account is never overridden."""
    close = {"a": _usage(50.0, 20.0), "b": _usage(45.0, 30.0)}
    name, reason = pick_account(["a", "b"], 0, 0, close)
    assert name == "a" and "within margin" in reason
    name, reason = pick_account(["a", "b"], 0, 0, {
        "a": _usage(90.0, 90.0),
        "b": None
    })
    assert name == "a" and "usage unavailable for b" in reason
    assert pick_account(["a"], 0, 3, {"a": None})[0] == "a"


def test_same_account_warning_names_identical_windows() -> None:
    """Two tokens of one account report the same windows; the warning names the
    pair and stays quiet otherwise."""
    warning = same_account_warning({
        "a": _usage(92.0, 84.0),
        "b": _usage(92.0, 84.0),
        "c": _usage(10.0, 84.0)
    })
    assert warning is not None and "a and b" in warning
    assert same_account_warning({"a": _usage(1.0, 2.0), "b": None}) is None


def test_account_block_usage_policy_picks_from_usage(tmp_path) -> None:
    """Under the usage policy the batch script's picker replaces the round-
    robin choice with the account that has the most budget left, and a picker
    that cannot read usage leaves the round-robin choice."""
    _write_token(tmp_path, "a", "sk-ant-oat01-AAA")
    _write_token(tmp_path, "b", "sk-ant-oat01-BBB")
    usage_file = tmp_path / "usage.json"
    usage_file.write_text(
        '{"a": {"session": 95, "weekly": 50}, '
        '"b": {"session": 20, "weekly": 30}}',
        encoding="utf-8")
    block = account_block(["a", "b"], offset=0, token_dir=tmp_path)
    repo_root = str(Path(__file__).resolve().parents[1])
    env = {
        POLICY_ENV_VAR: POLICY_USAGE,
        USAGE_FILE_ENV_VAR: str(usage_file),
        "PYTHONPATH": repo_root,
    }
    # Round-robin would give task 0 account a; usage says b.
    result = _run_account_block(block, 0, **env)
    assert result.returncode == 0, result.stderr
    assert "account=b token=sk-ant-oat01-BBB" in result.stdout
    assert "most budget left" in result.stderr
    # An unreadable account keeps round-robin (task 0 -> a).
    usage_file.write_text('{"a": {"session": 95, "weekly": 50}}',
                          encoding="utf-8")
    result = _run_account_block(block, 0, **env)
    assert result.returncode == 0, result.stderr
    assert "account=a token=sk-ant-oat01-AAA" in result.stdout
    assert "usage unavailable for b" in result.stderr


def test_accounts_cli_usage_and_pick(tmp_path, capsys, monkeypatch) -> None:
    """``usage`` prints the table and the same-account warning; ``pick`` prints
    just the account name on stdout."""
    _write_token(tmp_path, "a", "tok-a")
    _write_token(tmp_path, "b", "tok-b")
    usage_file = tmp_path / "usage.json"
    usage_file.write_text(
        '{"a": {"session": 92, "weekly": 84}, '
        '"b": {"session": 92, "weekly": 84}}',
        encoding="utf-8")
    monkeypatch.setenv(USAGE_FILE_ENV_VAR, str(usage_file))
    assert accounts_main(
        ["usage", "--accounts", "a,b", "--token-dir",
         str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "a" in out and "8%" in out and "same account" in out
    assert accounts_main([
        "pick", "--accounts", "a,b", "--offset", "1", "--task-id", "0",
        "--token-dir",
        str(tmp_path)
    ]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "b"
    assert "within margin" in captured.err


def test_read_usages_shares_fresh_readings_through_the_cache(
        tmp_path, monkeypatch) -> None:
    """Tasks starting together read one another's readings from the cache
    instead of each hitting the rate-limited endpoint; a stale entry is re-
    fetched, and an unreadable account is never cached."""
    _write_token(tmp_path, "a", "tok-a")
    _write_token(tmp_path, "b", "tok-b")
    monkeypatch.delenv(USAGE_FILE_ENV_VAR, raising=False)
    calls = []

    def fake_fetch(token: str, timeout: float = 0.0) -> AccountUsage:
        del timeout
        calls.append(token)
        if token == "tok-b":
            return None  # type: ignore[return-value]
        return _usage(session=40.0, weekly=10.0)

    monkeypatch.setattr(claude_accounts, "fetch_usage", fake_fetch)
    first = read_usages(["a", "b"], tmp_path)
    assert first["a"] == _usage(40.0, 10.0) and first["b"] is None
    assert calls == ["tok-a", "tok-b"]
    cache_path = tmp_path / ".usage-cache.json"
    assert cache_path.stat().st_mode & 0o077 == 0
    # A second reader within the cache window fetches only the missing one.
    second = read_usages(["a", "b"], tmp_path)
    assert second["a"] == _usage(40.0, 10.0)
    assert calls == ["tok-a", "tok-b", "tok-b"]
    # A stale entry is refreshed.
    stale = json.loads(cache_path.read_text(encoding="utf-8"))
    stale["a"]["at"] = 0.0
    cache_path.write_text(json.dumps(stale), encoding="utf-8")
    read_usages(["a"], tmp_path)
    assert calls[-1] == "tok-a"


def test_pick_account_keeps_off_a_limited_account() -> None:
    """A live limit marker on the round-robin choice moves the task onto a free
    account, round-robin over the free ones; with every account limited the
    plain round-robin choice stands; usage still decides among the free
    accounts."""
    usages = {"a": _usage(50.0, 20.0), "b": _usage(50.0, 20.0)}
    name, reason = pick_account(["a", "b"],
                                0,
                                0,
                                usages,
                                limited={
                                    "a": time.time() + 3600.0,
                                    "b": None
                                })
    assert name == "b" and "a is limited until" in reason
    name, reason = pick_account(["a", "b"],
                                0,
                                0,
                                usages,
                                limited={
                                    "a": 1.0e12,
                                    "b": 1.0e12
                                })
    assert name == "a" and "every account is limited" in reason
    three = {"a": _usage(50.0, 20.0), "b": _usage(90.0, 20.0), "c": None}
    name, reason = pick_account(["a", "b", "c"],
                                0,
                                0,
                                three,
                                limited={
                                    "a": None,
                                    "b": None,
                                    "c": 1.0e12
                                })
    assert name == "a" and "within margin" in reason
    three["b"] = _usage(5.0, 5.0)
    assert pick_account(["a", "b", "c"], 0, 0, three,
                        limited={"c": 1.0e12})[0] == "b"


def test_account_block_exports_the_limit_dir(tmp_path) -> None:
    """The batch script tells the harness where the markers live, under the
    launch's token directory."""
    _write_token(tmp_path, "a", "tok-a")
    block = account_block(["a"], offset=0, token_dir=tmp_path)
    result = _run_account_block(
        block + '\necho "limits=$PREDICATORS_CLAUDE_LIMIT_DIR"\n', 0)
    assert result.returncode == 0, result.stderr
    assert f"limits={tmp_path}/.limited" in result.stdout
