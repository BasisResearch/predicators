"""Agent runner for Docker sandbox.

Executed inside the Docker container by DockerSessionManager.  Loads a
pickled ``QueryInput`` dict, creates a ``ClaudeSDKClient`` session with
both Claude built-in tools (Bash, Read, Write, Edit, Glob, Grep) and
custom predicator MCP tools, queries the agent, and pickles results back
to a shared directory.

The predicators source tree is mounted read-only at ``/opt/predicators``
(via ``PYTHONPATH``) for imports.  Curated reference files are available
at ``/sandbox/reference/``.  A writable sandbox is at ``/sandbox``.
PreToolUse hooks restrict the agent's built-in tools to ``/sandbox/``.

Usage (inside Docker)::

    PYTHONPATH=/opt/predicators python3 \
        /opt/predicators/predicators/agent_sdk/docker_agent_runner.py \
        /data/query_input.pkl /data/query_output.pkl
"""
import asyncio
import json
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

# Bootstrap: import predicators.utils before anything else so that Python
# resolves the circular import chain (structs → utils → image_patch_wrapper
# → structs) in the correct order.  Without this, importing predicators.structs
# first causes image_patch_wrapper to try "from predicators.structs import Mask"
# while structs is still being initialized, raising an ImportError.
import predicators.utils  # noqa: F401, E402

import dill as pkl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Built-in Claude tools that the agent can use inside the container.
BUILTIN_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]

_MAX_PARAM_LEN = 120


def _truncate(value: Any, max_len: int = _MAX_PARAM_LEN) -> str:
    """Return a short string repr of *value*, truncating if needed."""
    s = repr(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


async def _run_query(query_input: Dict[str, Any]) -> Dict[str, Any]:
    """Create a ClaudeSDKClient, query the agent, and collect responses."""
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions,
                                  ClaudeSDKClient, ResultMessage, TextBlock,
                                  ToolResultBlock, ToolUseBlock, UserMessage,
                                  create_sdk_mcp_server)

    from predicators.agent_sdk.tools import (create_mcp_tools,
                                             get_allowed_tool_list)

    ctx = query_input["tool_context"]
    tool_names: Optional[List[str]] = query_input.get("tool_names")

    # Create MCP tools (closures over ctx — in-process, same as host)
    tools = create_mcp_tools(ctx, tool_names=tool_names)
    mcp_server = create_sdk_mcp_server(
        name="predicator_tools",
        version="1.0.0",
        tools=tools,
    )

    # Build allowed_tools: built-in Claude tools + custom MCP tools
    mcp_tool_list = get_allowed_tool_list(tool_names)
    allowed_tools = BUILTIN_TOOLS + mcp_tool_list

    options = ClaudeAgentOptions(
        allowed_tools=allowed_tools,
        mcp_servers={"predicator_tools": mcp_server},
        permission_mode="bypassPermissions",
        system_prompt=query_input["system_prompt"],
        model=query_input["model_name"],
        max_turns=query_input.get("max_turns", 20),
    )

    client = ClaudeSDKClient(options=options)
    await client.connect()

    collected: List[Dict[str, Any]] = []

    # Incremental log file path (on shared /data volume)
    log_path = query_input.get("log_path")

    def _flush_log() -> None:
        """Write current conversation state to the incremental log file."""
        if not log_path:
            return
        try:
            log_data = {
                "query": query_input.get("message", "")[:500],
                "response": collected,
            }
            with open(log_path, "w") as lf:
                json.dump(log_data, lf, indent=2, default=str)
        except Exception:
            pass  # Don't let logging errors break the agent

    try:
        await client.query(query_input["message"])
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                entry: Dict[str, Any] = {
                    "type": "assistant",
                    "content": [],
                }
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        entry["content"].append({
                            "type": "text",
                            "text": block.text,
                        })
                        print(f"Agent: {block.text[:200]}...",
                              file=sys.stderr, flush=True)
                    elif isinstance(block, ToolUseBlock):
                        entry["content"].append({
                            "type": "tool_use",
                            "id": getattr(block, "id", None),
                            "name": block.name,
                            "input": block.input,
                        })
                        # Summarise params (truncate long values)
                        params = block.input or {}
                        param_summary = ", ".join(
                            f"{k}={_truncate(v)}" for k, v in
                            params.items())
                        print(f"Tool call: {block.name}"
                              f"({param_summary})",
                              file=sys.stderr, flush=True)
                    else:
                        block_type = type(block).__name__
                        block_dict: Dict[str, Any] = {
                            "type": block_type,
                        }
                        for attr in ("name", "input", "id", "text",
                                     "content", "tool_use_id",
                                     "thinking", "signature"):
                            val = getattr(block, attr, None)
                            if val is not None:
                                block_dict[attr] = val
                        entry["content"].append(block_dict)
                        if block_type == "ThinkingBlock":
                            thinking = getattr(block, "thinking", "")
                            if thinking:
                                print(f"Thinking: {thinking[:200]}...",
                                      file=sys.stderr, flush=True)

                collected.append(entry)

            elif isinstance(msg, UserMessage):
                user_entry: Dict[str, Any] = {
                    "type": "user",
                    "content": [],
                }
                for block in msg.content:  # type: ignore[union-attr]
                    if isinstance(block, TextBlock):
                        user_entry["content"].append({
                            "type": "text",
                            "text": block.text,
                        })
                    elif isinstance(block, ToolResultBlock):
                        user_entry["content"].append({
                            "type":
                            "tool_result",
                            "tool_use_id":
                            getattr(block, "tool_use_id", None),
                            "content":
                            getattr(block, "content", None),
                            "is_error":
                            getattr(block, "is_error", False),
                        })
                    else:
                        block_dict2: Dict[str, Any] = {
                            "type": type(block).__name__,
                        }
                        for attr in ("name", "input", "id", "text",
                                     "content", "tool_use_id",
                                     "is_error"):
                            val = getattr(block, attr, None)
                            if val is not None:
                                block_dict2[attr] = val
                        user_entry["content"].append(block_dict2)
                collected.append(user_entry)

            elif isinstance(msg, ResultMessage):
                result_entry = {
                    "type": "result",
                    "num_turns": getattr(msg, "num_turns", None),
                    "total_cost_usd": getattr(msg, "total_cost_usd", None),
                }
                collected.append(result_entry)
                print(
                    f"Agent iteration complete. "
                    f"Turns: {getattr(msg, 'num_turns', '?')}, "
                    f"Cost: ${getattr(msg, 'total_cost_usd', '?')}",
                    file=sys.stderr, flush=True,
                )

            # Flush log after each message
            _flush_log()

    except Exception as e:
        logger.error("Agent session error: %s", e)
        collected.append({"type": "error", "error": str(e)})
        _flush_log()
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    return {
        "responses": collected,
        "iteration_proposals": ctx.iteration_proposals,
    }


def _rehash_objects_after_unpickle(ctx: Any) -> None:
    """Fix stale Object hash caches after cross-process unpickling.

    ``Object.__hash__`` returns a ``cached_property`` (``_hash``) that
    stores ``hash(str(self))``.  Python randomises string hashes across
    processes (PYTHONHASHSEED), so cached values from the *pickling*
    process are stale here.  When the option-model simulator later
    creates fresh Objects (e.g. ``self._robot`` in ``_get_state``),
    their hashes differ from the unpickled Objects, causing KeyError on
    ``State.data`` dict lookups.

    Fix: clear every Object's cached ``_hash`` (and ``_str``) so it is
    re-computed with the current process's hash seed, then rebuild every
    ``State.data`` dict so its internal hash-table is consistent.
    """
    from predicators.structs import State

    seen: set = set()

    def _clear(obj: Any) -> None:
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        obj.__dict__.pop("_hash", None)
        obj.__dict__.pop("_str", None)

    def _process_state(state: Any) -> None:
        if state is None or not isinstance(state, State):
            return
        for obj in list(state.data.keys()):
            _clear(obj)
        # Rebuild dict so Python re-hashes keys with current seed.
        state.data = {k: v for k, v in state.data.items()}

    def _process_atoms(atoms: Any) -> None:
        for atom in atoms:
            for obj in atom.objects:
                _clear(obj)

    def _process_task(task: Any) -> None:
        # Task has .init (State) and .goal (Set[GroundAtom])
        # EnvironmentTask has .init_obs and .goal_description
        if hasattr(task, "init"):
            _process_state(task.init)
        if hasattr(task, "init_obs"):
            _process_state(task.init_obs)
        for attr in ("goal", "alt_goal", "goal_description",
                      "alt_goal_desc"):
            atoms = getattr(task, attr, None)
            if atoms:
                _process_atoms(atoms)

    # Train tasks
    for task in getattr(ctx, "train_tasks", []):
        _process_task(task)

    # Current task
    if ctx.current_task is not None:
        _process_task(ctx.current_task)

    # Example state
    _process_state(getattr(ctx, "example_state", None))

    # Trajectories
    for traj in (getattr(ctx, "offline_trajectories", []) +
                 getattr(ctx, "online_trajectories", [])):
        for state in traj.states:
            _process_state(state)


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pkl> <output.pkl>",
              file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    logger.info("Docker agent runner starting: input=%s output=%s",
                input_path, output_path)

    # Load query input
    with open(input_path, "rb") as f:
        query_input = pkl.load(f)

    # Restore host CFG settings (arg-specific settings like
    # max_num_steps_option_rollout are not set by default import)
    if "cfg_snapshot" in query_input:
        from predicators.settings import CFG
        for k, v in query_input["cfg_snapshot"].items():
            setattr(CFG, k, v)

    # Fix stale Object hash caches from cross-process pickling.
    ctx = query_input.get("tool_context")
    if ctx is not None:
        _rehash_objects_after_unpickle(ctx)

    # Recreate option model — the simulator (e.g. PyBullet physics
    # server) is process-local and cannot survive pickling.
    if ctx is not None and ctx.option_model is not None:
        from predicators.option_model import create_option_model
        from predicators.settings import CFG as _cfg
        logger.info("Recreating option model (%s) inside Docker...",
                    _cfg.option_model_name)
        ctx.option_model = create_option_model(_cfg.option_model_name)

    logger.info("Loaded query input: message length=%d, model=%s",
                len(query_input.get("message", "")),
                query_input.get("model_name", "?"))

    # Run the query
    try:
        query_output = asyncio.run(_run_query(query_input))
    except Exception as e:
        logger.error("Fatal error in agent runner: %s\n%s", e,
                     traceback.format_exc())
        query_output = {
            "responses": [{
                "type": "error",
                "error": str(e)
            }],
            "iteration_proposals": None,
        }

    # Save output
    with open(output_path, "wb") as f:
        pkl.dump(query_output, f)

    logger.info("Docker agent runner finished: %d responses",
                len(query_output.get("responses", [])))


if __name__ == "__main__":
    main()
