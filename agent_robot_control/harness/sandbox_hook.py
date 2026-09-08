#!/usr/bin/env python3
"""Claude Code PreToolUse hook: keep the agent inside its workspace.

Denies Read/Write/Edit/Glob/Grep whose path resolves outside the workspace
(the hook's cwd), and Bash commands that mention the predicators repo, the
user's Claude config, `..`, or absolute paths outside the workspace. The
agent must learn about the scene through the robot tools, not by reading the
environment source code or this project's notes.

Configured through a per-run settings.json (see claude_code.py). Reads the
hook payload on stdin; prints a permission decision on stdout.
"""
import json
import os
import re
import sys


def deny(reason: str) -> None:
    json.dump({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                      "permissionDecision": "deny",
                                      "permissionDecisionReason": reason}},
              sys.stdout)
    sys.exit(0)


def main() -> None:
    data = json.load(sys.stdin)
    tool = data.get("tool_name", "")
    inp = data.get("tool_input", {}) or {}
    workspace = os.path.realpath(os.environ.get("ARC_WORKSPACE") or os.getcwd())
    blocked_substrings = [
        s for s in os.environ.get("ARC_BLOCKED_PATHS", "").split(os.pathsep) if s
    ]

    def outside(path: str) -> bool:
        resolved = os.path.realpath(os.path.join(workspace, os.path.expanduser(path)))
        return not (resolved == workspace or resolved.startswith(workspace + os.sep))

    if tool in ("Read", "Write", "Edit", "NotebookEdit"):
        path = inp.get("file_path") or inp.get("notebook_path") or ""
        if path and outside(path):
            deny(f"Blocked: {path} is outside the workspace {workspace}. Only the "
                 "workspace is accessible; use the robot tools to learn about the scene.")
    elif tool in ("Glob", "Grep"):
        path = inp.get("path") or ""
        if path and outside(path):
            deny(f"Blocked: {path} is outside the workspace {workspace}.")
    elif tool == "Bash":
        cmd = inp.get("command", "")
        for s in blocked_substrings:
            if s and s in cmd:
                deny(f"Blocked: commands may not reference {s}.")
        if ".." in cmd or "~/.claude" in cmd or "$HOME" in cmd:
            deny("Blocked: commands may not leave the workspace.")
        for m in re.findall(r"(?<![\w.])(/[\w./-]+)", cmd):
            if m.startswith(("/usr", "/bin", "/dev", "/proc", "/tmp", "/etc")):
                continue
            if outside(m):
                deny(f"Blocked: {m} is outside the workspace {workspace}.")
    sys.exit(0)


if __name__ == "__main__":
    main()
