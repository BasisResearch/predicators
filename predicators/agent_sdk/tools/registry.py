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
# Solve-journal writing (agent_solve_use_journal): agent-authored
# lessons for future fresh-context attempts. Read side is prompt
# injection, so this is the only journal tool.
JOURNAL_TOOL_NAMES = [
    "record_journal",
]

ALL_TOOL_NAMES = (TESTING_TOOL_NAMES + EXPLORATION_TOOL_NAMES +
                  JOURNAL_TOOL_NAMES)

# Names of tools returned by ``create_synthesis_tools`` (sim-learning)
# and ``create_predicate_synthesis_tools`` (predicate invention). These
# tools are produced by ``AgentSessionMixin._build_synthesis_mcp_tools``
# and joined to the static MCP set at session-open time; the constants
# exist so callers / tests can refer to them without typing the strings
# twice. ``tests/agent_sdk/test_tool_registry.py`` asserts that the
# factory outputs match these tuples.
SYNTHESIS_TOOL_NAMES = ("run_python", )
PREDICATE_SYNTHESIS_TOOL_NAMES = ("evaluate_predicate_quality", )
SAMPLER_SYNTHESIS_TOOL_NAMES = ("evaluate_sampler", )


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
