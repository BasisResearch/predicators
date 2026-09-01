"""Assembly of the static MCP toolset (create_mcp_tools)."""
from typing import List, Optional

from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.exploration import _build_exploration_tools
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result
from predicators.agent_sdk.tools.testing import _build_testing_tools


def create_mcp_tools(ctx: ToolContext,
                     tool_names: Optional[List[str]] = None) -> list:
    """Create MCP tools with the given ToolContext via closures.

    Args:
        ctx: Shared mutable state between the approach and MCP tools.
        tool_names: If provided, only return tools with these names.
            If None, return all tools.

    Returns a list of SdkMcpTool objects to pass to create_sdk_mcp_server.
    """
    from claude_agent_sdk import \
        tool as _sdk_tool  # pylint: disable=import-outside-toplevel
    tool = _make_coercing_tool(_sdk_tool)

    # Spill oversize tool output into the sandbox (``./tool_outputs/``)
    # instead of returning it inline. Each builder names its parameter
    # ``_text_result`` so every nested tool's ``_text_result(...)`` call
    # routes through the spiller with no call-site edits.
    _text_result = _make_spilling_text_result(ctx.sandbox_dir)

    # A session-specific instance attached to ``ctx.extra_mcp_tools``
    # wins over the static builder of the same name: synthesis sessions
    # attach their own ``run_python`` (fit data + candidate-simulator
    # probe in one namespace), so the solve-phase instance is neither
    # built nor offered there.
    extra_names = {getattr(t, "name", "") for t in ctx.extra_mcp_tools}
    _all = {
        **_build_testing_tools(ctx, _text_result, tool),
        **({} if "run_python" in extra_names else _build_exploration_tools(
               ctx, _text_result, tool)),
    }
    if tool_names is None:
        tools = list(_all.values())
    else:
        tools = [_all[n] for n in tool_names if n in _all]
    tools = [t for t in tools if getattr(t, "name", "") not in extra_names]
    tools.extend(ctx.extra_mcp_tools)
    return tools
