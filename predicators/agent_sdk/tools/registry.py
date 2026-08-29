"""Tool-name rosters and the session tool-list surface."""
from typing import Any, Dict, List, Optional, Sequence

MCP_SERVER_NAME = "predicator_tools"

# Built-in Claude tools available to the sandboxed agent.
BUILTIN_TOOLS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Task",
    "TaskOutput",
    "TaskStop",
    "TaskCreate",
    "TaskGet",
    "TaskUpdate",
    "TaskList",
]

TESTING_TOOL_NAMES = [
    "evaluate_option_plan",
    # Closed-loop policy mode (agent_solve_policy_mode): validates and
    # captures the agent-written policy.py. Only offered on solve
    # rosters when the mode is on.
    "evaluate_policy",
]
# The one code-execution tool. Solve sessions get the static instance
# built by ``create_mcp_tools`` (namespace = the BeliefProbe facade over
# the deployed belief model, predicators/agent_sdk/belief_probe.py);
# synthesis sessions attach their own instance under the same name
# (fit data + the probe over the candidate simulator), which replaces
# the static one at assembly. Offered to every session that has a
# simulator to probe (see ``AgentModelFreeApproach._get_solve_tool_names``).
EXPLORATION_TOOL_NAMES = [
    "run_python",
]
ALL_TOOL_NAMES = TESTING_TOOL_NAMES + EXPLORATION_TOOL_NAMES

# Name of the tool ``create_synthesis_tools`` builds for a synthesis
# session (the same ``run_python`` name as the solve-phase instance -
# see EXPLORATION_TOOL_NAMES). ``tests/agent_sdk/test_tool_registry.py``
# asserts that the factory output matches this tuple. Predicate and
# sampler drafts are loaded through the probe (``sim.predicates()`` /
# ``sim.samplers()``), not through tools.
SYNTHESIS_TOOL_NAMES = ("run_python", )


def get_allowed_tool_list(tool_names: Optional[List[str]] = None) -> List[str]:
    """Compute the allowed_tools list for the agent SDK.

    ``tool_names`` is the caller's declared tool surface; it may mix
    static MCP names (in ``ALL_TOOL_NAMES``) with names of dynamic
    ``SdkMcpTool`` instances supplied via ``ctx.extra_mcp_tools``. We do
    not silently filter — typos surface as "unknown tool" errors from
    the SDK rather than as missing-allowlist mysteries. Passing ``None``
    keeps the legacy "all static MCP tools" default.
    """
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    names = list(ALL_TOOL_NAMES) if tool_names is None else list(tool_names)
    return [f"{prefix}{n}" for n in names]


def list_session_tool_names(
    *,
    mcp_filter: Optional[Sequence[str]] = None,
    extra_mcp_tools: Sequence[Any] = (),
    include_builtin: bool = True,
) -> Dict[str, List[str]]:
    """Return the tool names active in a session, grouped by source.

    A convenience view of "what does this agent session see?" — useful
    for logs and prompt-construction debugging. Names are bare (no
    ``mcp__predicator_tools__`` prefix); use ``get_allowed_tool_list``
    for the prefixed form Claude Agent SDK expects.

    Args:
        mcp_filter: Subset of ``ALL_TOOL_NAMES`` to keep. ``None`` (the
            default) lists every MCP tool.
        extra_mcp_tools: Synthesis tools supplied for the session
            (e.g. by ``_build_synthesis_mcp_tools``). Their names are
            read off each tool's ``name`` attribute.
        include_builtin: Whether to include the Claude built-in tools
            (``Bash``, ``Read``, ``Write``, …).

    Returns ``{"builtin": [...], "mcp": [...], "extra": [...]}``.
    """
    valid = set(ALL_TOOL_NAMES)
    if mcp_filter is None:
        mcp_names = list(ALL_TOOL_NAMES)
    else:
        mcp_names = [n for n in mcp_filter if n in valid]
    extra_names = [
        getattr(t, "name", "") for t in extra_mcp_tools
        if getattr(t, "name", "")
    ]
    out: Dict[str, List[str]] = {"mcp": mcp_names, "extra": extra_names}
    if include_builtin:
        out["builtin"] = list(BUILTIN_TOOLS)
    return out
