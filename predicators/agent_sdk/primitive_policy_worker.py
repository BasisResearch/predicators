"""JSON worker for agent-authored low-level policies.

Runs in a fresh interpreter under the sandbox's Python guard. Only JSON
observations cross the process boundary; no environment, task evaluator,
skill, State object or live simulator handle is supplied to policy code.
"""
import contextlib
import json
import os
import runpy
import sys
from typing import Any, Dict

REPLY_PREFIX = "PREDICATORS_POLICY_REPLY:"


def main() -> None:
    """Load one policy and retain its globals and memory until stdin closes."""
    output = sys.stdout
    memory: Dict[str, Any] = {}
    sys.path.insert(0, os.getcwd())

    def reply(payload: Dict[str, Any]) -> None:
        output.write(REPLY_PREFIX + json.dumps(payload, allow_nan=False) +
                     "\n")
        output.flush()

    with contextlib.redirect_stdout(sys.stderr):
        try:
            namespace = runpy.run_path(sys.argv[1])
            policy = namespace.get("get_action")
            if not callable(policy):
                raise ValueError("policy must define get_action(observation, "
                                 "memory)")
            reply({"ready": True})
            for line in sys.stdin:
                observation = json.loads(line)
                action = policy(observation, memory)
                if action is not None:
                    action = [float(value) for value in action]
                reply({"action": action})
        except Exception as err:  # pylint: disable=broad-except
            reply({"error": f"{type(err).__name__}: {err}"})


if __name__ == "__main__":
    main()
