"""Spread a launch's runs over several Claude accounts.

Usage limits are per account (rolling five-hour windows plus a weekly
cap), so two accounts sustain about twice the concurrent agent runs
before the harness has to sleep out a limit. Each account is a
long-lived OAuth token stored in its own file:

    ~/.claude-tokens/<name>      # mode 600, holding one token

minted with ``CLAUDE_CONFIG_DIR=~/.claude-<name> claude setup-token``
(the config dir keeps the mint from touching the CLI's own login).

The reserved name ``login`` is the CLI's own stored login (whatever
``claude /login`` last signed into), which needs no token file.

A launch names its accounts once (``--accounts a,b`` or the
``PREDICATORS_CLAUDE_ACCOUNTS`` environment variable). Every array task
(one per seed) of every experiment in the config then picks its account
when it starts, under one of two policies
(``PREDICATORS_CLAUDE_ACCOUNT_POLICY``):

``usage`` (the default)
    The task reads every account's usage windows from the OAuth usage
    endpoint and takes the account with the most budget left, where an
    account's leftover is what remains of its tightest window (session
    or weekly). When the leftover of the best account is within
    ``PICK_MARGIN`` points of the round-robin choice below, the
    round-robin choice wins, so a burst of tasks starting together still
    spreads instead of piling onto one account. When any account's
    usage cannot be read, the round-robin choice is used.

``round_robin``
    Account ``(experiment index + seed) mod n``, so a config's runs
    spread evenly even when it holds a single arm with many seeds, and
    a requeued task lands on the account it started on.

The batch script reads the token from its file at run time rather than
carrying it, so nothing secret is written into the script or the Slurm
environment, and a rotated token takes effect on the next (re)start.

The chosen account is exported to the run as ``PREDICATORS_CLAUDE_ACCOUNT``
and recorded on the scorecard (``claude_account``), so per-account spend
can be read off the aggregated results.

Two token files that belong to the same account report identical usage,
and a launch warns about it: such a pair spreads nothing.

Command line (also what the batch script calls)::

    python -m scripts.engaging.claude_accounts usage --accounts a,b
    python -m scripts.engaging.claude_accounts pick --accounts a,b \\
        --offset 0 --task-id 1
"""

import argparse
import json
import os
import re
import shlex
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

TOKEN_DIR = Path.home() / ".claude-tokens"
LOGIN_ACCOUNT = "login"
# Launch-time override for callers that do not pass --accounts.
ACCOUNTS_ENV_VAR = "PREDICATORS_CLAUDE_ACCOUNTS"
# Exported into each job; predicators/run/scorecard.py records it.
ACCOUNT_ENV_VAR = "PREDICATORS_CLAUDE_ACCOUNT"
# The variable the Claude CLI reads its OAuth token from.
TOKEN_ENV_VAR = "CLAUDE_CODE_OAUTH_TOKEN"
# How a task picks its account: "usage" (default) or "round_robin".
POLICY_ENV_VAR = "PREDICATORS_CLAUDE_ACCOUNT_POLICY"
POLICY_USAGE = "usage"
POLICY_ROUND_ROBIN = "round_robin"
POLICIES = (POLICY_USAGE, POLICY_ROUND_ROBIN)
# A JSON file of ``{account: {"session": pct, "weekly": pct}}`` that
# stands in for the usage endpoint (tests, or an offline dry run).
USAGE_FILE_ENV_VAR = "PREDICATORS_CLAUDE_USAGE_FILE"
# The usage endpoint the Claude CLI's /usage reads; the beta header is
# what admits an OAuth bearer token there.
USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
USAGE_BETA_HEADER = "oauth-2025-04-20"
USAGE_TIMEOUT_SECS = 15.0
# Leftover-budget lead (percentage points) the best account needs over
# the round-robin choice before it displaces it.
PICK_MARGIN = 10.0

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def token_path(account: str, token_dir: Path = TOKEN_DIR) -> Path:
    """Where ``account``'s token file lives."""
    return token_dir / account


def _check_token_file(account: str, path: Path) -> None:
    if not path.is_file():
        raise ValueError(
            f"Claude account {account!r}: no token file at {path}. Mint one "
            f"with `CLAUDE_CONFIG_DIR=~/.claude-{account} claude setup-token` "
            f"and save the printed token there with mode 600.")
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ValueError(
            f"Claude account {account!r}: {path} is readable by others "
            f"(mode {mode:03o}); run `chmod 600 {path}`.")
    if not path.read_text(encoding="utf-8").strip():
        raise ValueError(f"Claude account {account!r}: {path} is empty.")


def resolve_accounts(accounts: Optional[str],
                     token_dir: Path = TOKEN_DIR) -> List[str]:
    """The account list for a launch: the explicit comma-separated argument,
    then the ``PREDICATORS_CLAUDE_ACCOUNTS`` environment variable, then just
    the CLI's stored login.

    Every named account's token file is checked up front (exists, is
    private, is non-empty), so a bad name fails the launch rather than
    the job.
    """
    spec = accounts if accounts is not None else os.environ.get(
        ACCOUNTS_ENV_VAR, "")
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names:
        return [LOGIN_ACCOUNT]
    seen = set()
    for name in names:
        if not _NAME_RE.match(name):
            raise ValueError(f"Claude account name {name!r} is not a plain "
                             "file name (letters, digits, '_', '.', '-').")
        if name in seen:
            raise ValueError(f"Claude account {name!r} is listed twice.")
        seen.add(name)
        if name != LOGIN_ACCOUNT:
            _check_token_file(name, token_path(name, token_dir))
    return names


def account_policy() -> str:
    """The pick policy from ``PREDICATORS_CLAUDE_ACCOUNT_POLICY``; the default
    is ``usage``."""
    policy = os.environ.get(POLICY_ENV_VAR, POLICY_USAGE).strip().lower()
    if policy not in POLICIES:
        raise ValueError(f"{POLICY_ENV_VAR}={policy!r}: expected one of "
                         f"{', '.join(POLICIES)}.")
    return policy


def assigned_account(accounts: Sequence[str], offset: int,
                     task_id: int) -> str:
    """The round-robin account of array task ``task_id`` (its seed) of the
    experiment at ``offset`` in the launch config; mirrors the batch script's
    fallback."""
    return accounts[(offset + task_id) % len(accounts)]


def describe_assignment(accounts: Sequence[str], offset: int, start_seed: int,
                        num_seeds: int) -> str:
    """One line naming each account's round-robin seeds, for the launch log."""
    seeds_by_account: Dict[str, List[int]] = {name: [] for name in accounts}
    for seed in range(start_seed, start_seed + num_seeds):
        seeds_by_account[assigned_account(accounts, offset, seed)].append(seed)
    parts = [
        f"{name} <- seeds {','.join(str(s) for s in seeds)}"
        for name, seeds in seeds_by_account.items() if seeds
    ]
    return "; ".join(parts)


# -- Usage windows ----------------------------------------------------------


@dataclass(frozen=True)
class AccountUsage:
    """Utilization (percent) of an account's two limit windows."""

    session: float
    weekly: float

    @property
    def leftover(self) -> float:
        """Budget left in the tightest window, in percentage points."""
        return 100.0 - max(self.session, self.weekly)


def account_token(account: str, token_dir: Path = TOKEN_DIR) -> Optional[str]:
    """The bearer token that reads ``account``'s usage: the token file, or for
    ``login`` the CLI's stored (unexpired) access token."""
    if account != LOGIN_ACCOUNT:
        path = token_path(account, token_dir)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8").strip() or None
    config_dir = Path(
        os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude")))
    creds_path = config_dir / ".credentials.json"
    if not creds_path.is_file():
        return None
    try:
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
        oauth = creds["claudeAiOauth"]
        if float(oauth.get("expiresAt", 0)) / 1000.0 <= time.time():
            return None
        return str(oauth["accessToken"]) or None
    except (ValueError, KeyError, TypeError):
        return None


def fetch_usage(token: str,
                timeout: float = USAGE_TIMEOUT_SECS) -> Optional[AccountUsage]:
    """Read one account's usage windows; None when the endpoint refuses or
    cannot be reached (the caller falls back to round-robin)."""
    request = urllib.request.Request(USAGE_URL,
                                     headers={
                                         "Authorization": f"Bearer {token}",
                                         "anthropic-beta": USAGE_BETA_HEADER,
                                         "Accept": "application/json",
                                     })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return AccountUsage(session=float(body["five_hour"]["utilization"]),
                            weekly=float(body["seven_day"]["utilization"]))
    except (urllib.error.URLError, OSError, ValueError, KeyError,
            TypeError) as exc:
        print(f"[claude-account] usage unavailable: {exc}", file=sys.stderr)
        return None


def read_usages(
        accounts: Sequence[str],
        token_dir: Path = TOKEN_DIR) -> Dict[str, Optional[AccountUsage]]:
    """Usage of every account, from ``PREDICATORS_CLAUDE_USAGE_FILE`` when set,
    else from the usage endpoint."""
    usage_file = os.environ.get(USAGE_FILE_ENV_VAR)
    usages: Dict[str, Optional[AccountUsage]] = {}
    if usage_file:
        with open(usage_file, encoding="utf-8") as f:
            table = json.load(f)
        for name in accounts:
            entry = table.get(name)
            usages[name] = (None if entry is None else
                            AccountUsage(session=float(entry["session"]),
                                         weekly=float(entry["weekly"])))
        return usages
    for name in accounts:
        token = account_token(name, token_dir)
        usages[name] = None if token is None else fetch_usage(token)
    return usages


def pick_account(accounts: Sequence[str],
                 offset: int,
                 task_id: int,
                 usages: Mapping[str, Optional[AccountUsage]],
                 margin: float = PICK_MARGIN) -> Tuple[str, str]:
    """The account a task should run on and a one-line reason.

    The account with the most leftover budget wins when it leads the
    round-robin choice by at least ``margin`` points; otherwise the
    round-robin choice stands (a burst of simultaneous starts sees the
    same snapshot, and the margin keeps it spread). Any unreadable usage
    means round-robin.
    """
    fallback = assigned_account(accounts, offset, task_id)
    if len(accounts) == 1:
        return fallback, "single account"
    known = {
        name: usage
        for name, usage in usages.items() if usage is not None
    }
    missing = [name for name in accounts if name not in known]
    if missing:
        return fallback, ("round-robin: usage unavailable for "
                          f"{', '.join(missing)}")
    leftovers = {name: known[name].leftover for name in accounts}
    best = max(accounts, key=lambda name: leftovers[name])
    summary = ", ".join(f"{name} {leftovers[name]:.0f}%" for name in accounts)
    if best == fallback or leftovers[best] - leftovers[fallback] < margin:
        return fallback, f"round-robin within margin ({summary} left)"
    return best, f"most budget left ({summary} left)"


def same_account_warning(
        usages: Mapping[str, Optional[AccountUsage]]) -> Optional[str]:
    """A warning naming account pairs whose usage windows are identical, which
    is what two tokens of one account look like."""
    seen: Dict[Tuple[float, float], str] = {}
    pairs = []
    for name, usage in usages.items():
        if usage is None:
            continue
        key = (usage.session, usage.weekly)
        if key in seen:
            pairs.append(f"{seen[key]} and {name}")
        else:
            seen[key] = name
    if not pairs:
        return None
    return ("[claude-account] WARNING: identical usage windows for "
            f"{'; '.join(pairs)}: probably tokens of the same account, "
            "which spreads no load.")


def format_usages(accounts: Sequence[str],
                  usages: Mapping[str, Optional[AccountUsage]]) -> str:
    """A small table of session/weekly utilization and leftover budget."""
    lines = [f"{'account':<10} {'session':>8} {'weekly':>8} {'left':>6}"]
    for name in accounts:
        usage = usages.get(name)
        if usage is None:
            lines.append(f"{name:<10} {'?':>8} {'?':>8} {'?':>6}")
        else:
            lines.append(f"{name:<10} {usage.session:>7.0f}% "
                         f"{usage.weekly:>7.0f}% {usage.leftover:>5.0f}%")
    return "\n".join(lines)


# -- The batch-script block -------------------------------------------------


def account_block(accounts: Sequence[str],
                  offset: int,
                  token_dir: Path = TOKEN_DIR) -> str:
    """Bash that picks this array task's account and exports its token.

    The round-robin choice is computed in bash; under the ``usage``
    policy the picker (``python -m scripts.engaging.claude_accounts
    pick``) may replace it, and any picker failure keeps the round-robin
    choice. A missing or empty token file ends the job with a clear
    error instead of silently running on the CLI's stored login.
    """
    assert accounts, "an account list is never empty"
    names = " ".join(shlex.quote(name) for name in accounts)
    spec = shlex.quote(",".join(accounts))
    dir_arg = shlex.quote(str(token_dir))
    return f"""\
# Claude account for this array task: round-robin over the launch's
# accounts by (experiment index + seed), or under the usage policy the
# account with the most budget left (claude_accounts.py). The token is
# read from its file here, at run time, so it never appears in this
# script.
_ACCOUNTS=({names})
_ACCOUNT="${{_ACCOUNTS[$(( ({offset} + ${{SLURM_ARRAY_TASK_ID:-0}}) % {len(accounts)} ))]}}"
if [ "${{{POLICY_ENV_VAR}:-{POLICY_USAGE}}}" = "{POLICY_USAGE}" ]; then
  _PICKED="$(python -m scripts.engaging.claude_accounts pick --accounts {spec} \\
    --offset {offset} --task-id "${{SLURM_ARRAY_TASK_ID:-0}}" --token-dir {dir_arg})"
  if [ -n "$_PICKED" ]; then _ACCOUNT="$_PICKED"; fi
fi
if [ "$_ACCOUNT" = "{LOGIN_ACCOUNT}" ]; then
  unset {TOKEN_ENV_VAR}
else
  _TOKEN_FILE={dir_arg}/"$_ACCOUNT"
  {TOKEN_ENV_VAR}="$(tr -d '[:space:]' < "$_TOKEN_FILE")"
  if [ -z "${TOKEN_ENV_VAR}" ]; then
    echo "[claude-account] no usable token in $_TOKEN_FILE" >&2
    exit 1
  fi
  export {TOKEN_ENV_VAR}
fi
export {ACCOUNT_ENV_VAR}="$_ACCOUNT"
echo "[claude-account] $_ACCOUNT"
"""


# -- Command line -----------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.engaging.claude_accounts",
        description="Read Claude account usage or pick a task's account.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    usage = subparsers.add_parser("usage", help="print each account's usage")
    pick = subparsers.add_parser(
        "pick", help="print the account an array task should run on")
    for sub in (usage, pick):
        sub.add_argument("--accounts",
                         required=True,
                         help="comma-separated account names")
        sub.add_argument("--token-dir", type=Path, default=TOKEN_DIR)
    pick.add_argument("--offset", type=int, required=True)
    pick.add_argument("--task-id", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point; ``pick`` never fails the caller: on any problem it prints
    nothing and the batch script keeps its round-robin choice."""
    args = _parse_args(argv)
    accounts = [n.strip() for n in args.accounts.split(",") if n.strip()]
    if args.command == "usage":
        usages = read_usages(accounts, args.token_dir)
        print(format_usages(accounts, usages))
        warning = same_account_warning(usages)
        if warning:
            print(warning)
        return 0
    try:
        usages = read_usages(accounts, args.token_dir)
        name, reason = pick_account(accounts, args.offset, args.task_id,
                                    usages)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[claude-account] pick failed, keeping round-robin: {exc}",
              file=sys.stderr)
        return 0
    print(f"[claude-account] pick: {name} ({reason})", file=sys.stderr)
    warning = same_account_warning(usages)
    if warning:
        print(warning, file=sys.stderr)
    print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
