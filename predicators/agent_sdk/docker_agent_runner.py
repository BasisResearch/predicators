"""Agent runner for Docker sandbox.

Executed inside the Docker container by DockerSessionManager.  Loads a
pickled ``QueryInput`` dict, creates a ``ClaudeSDKClient`` session with
both Claude built-in tools (Bash, Read, Write, Edit, Glob, Grep) and
custom predicator MCP tools, queries the agent, and pickles results back
to a shared directory.

The predicators source tree is mounted read-only at ``/workspace`` and
can be inspected by the agent via built-in tools.  A writable sandbox
is available at ``/sandbox``.

Usage (inside Docker)::

    PYTHONPATH=/workspace python3 \
        /workspace/predicators/agent_sdk/docker_agent_runner.py \
        /data/query_input.pkl /data/query_output.pkl
"""
import asyncio
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional

import dill as pkl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Built-in Claude tools that the agent can use inside the container.
BUILTIN_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]


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
                        logger.debug("Agent: %s...", block.text[:200])
                    elif isinstance(block, ToolUseBlock):
                        entry["content"].append({
                            "type": "tool_use",
                            "id": getattr(block, "id", None),
                            "name": block.name,
                            "input": block.input,
                        })
                        logger.debug("Agent tool call: %s", block.name)
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

            elif isinstance(msg, ResultMessage):
                result_entry = {
                    "type": "result",
                    "num_turns": getattr(msg, "num_turns", None),
                    "total_cost_usd": getattr(msg, "total_cost_usd", None),
                }
                collected.append(result_entry)
                logger.info(
                    "Agent iteration complete. Turns: %s, Cost: $%s",
                    getattr(msg, "num_turns", "?"),
                    getattr(msg, "total_cost_usd", "?"),
                )
    except Exception as e:
        logger.error("Agent session error: %s", e)
        collected.append({"type": "error", "error": str(e)})
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    return {
        "responses": collected,
        "iteration_proposals": ctx.iteration_proposals,
    }


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
