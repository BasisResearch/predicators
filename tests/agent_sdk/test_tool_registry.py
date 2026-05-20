"""Smoke tests for the agent-SDK tool registry.

Guards against drift between the ``@tool("name", ...)`` decorators
inside the factory functions and the name tuples exported from
``predicators.agent_sdk.tools``.  If a new tool is added (or renamed)
without updating the constants, these tests fail.
"""
# pylint: disable=protected-access
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Set

from predicators.agent_sdk.agent_session_mixin import AgentSessionMixin
from predicators.agent_sdk.tools import ALL_TOOL_NAMES, BUILTIN_TOOLS, \
    MCP_SERVER_NAME, PREDICATE_SYNTHESIS_TOOL_NAMES, SYNTHESIS_TOOL_NAMES, \
    ToolContext, create_mcp_tools, create_predicate_synthesis_tools, \
    create_synthesis_tools, get_allowed_tool_list, list_session_tool_names


def _names(tools: Iterable[Any]) -> Set[str]:
    return {getattr(t, "name", "") for t in tools}


def test_create_mcp_tools_matches_all_tool_names() -> None:
    """``create_mcp_tools`` exposes exactly the names in ``ALL_TOOL_NAMES``."""
    tools = create_mcp_tools(ToolContext())
    assert _names(tools) == set(ALL_TOOL_NAMES)


def test_create_synthesis_tools_matches_constant(tmp_path) -> None:
    """``create_synthesis_tools`` builds exactly the synthesis name tuple."""
    tools = create_synthesis_tools(
        exec_ns={},
        base_pred_triples=[],
        inferred_process_features={},
        simulator_file=str(tmp_path / "simulator.py"),
        versions_dir=str(tmp_path / "simulator_versions"),
        approach=None,
    )
    assert _names(tools) == set(SYNTHESIS_TOOL_NAMES)


def test_create_predicate_synthesis_tools_matches_constant(tmp_path) -> None:
    """Predicate-synthesis builder matches the predicate-synthesis name
    tuple."""
    approach_stub = SimpleNamespace(_fitted_params={})
    tools = create_predicate_synthesis_tools(
        predicates_file=str(tmp_path / "predicates.py"),
        predicates_versions_dir=str(tmp_path / "predicates_versions"),
        approach=approach_stub,
        trajectories=[],
    )
    assert _names(tools) == set(PREDICATE_SYNTHESIS_TOOL_NAMES)


def test_list_session_tool_names_defaults() -> None:
    """Default ``list_session_tool_names`` returns all MCP + builtin tools."""
    grouped = list_session_tool_names()
    assert grouped["mcp"] == list(ALL_TOOL_NAMES)
    assert grouped["extra"] == []
    assert grouped["builtin"] == list(BUILTIN_TOOLS)


def test_list_session_tool_names_filters_and_combines() -> None:
    """Filtered MCP names drop unknowns; ``extra_mcp_tools`` pass through."""
    fake = SimpleNamespace(name="run_python")
    grouped = list_session_tool_names(
        mcp_filter=["inspect_options", "not_a_tool", "annotate_scene"],
        extra_mcp_tools=[fake],
        include_builtin=False,
    )
    assert grouped == {
        "mcp": ["inspect_options", "annotate_scene"],
        "extra": ["run_python"],
    }


def test_synthesis_tool_names_default_is_empty() -> None:
    """No synthesis MCP filter by default — approaches with no synthesis phase
    get an empty allowlist for free."""
    obj = AgentSessionMixin()
    assert not obj._get_synthesis_tool_names()


def test_solve_and_synthesis_tool_names_are_independent() -> None:
    """Subclasses can declare disjoint solve / synthesis tool sets."""

    # pylint: disable=abstract-method
    class _Approach(AgentSessionMixin):

        def _get_solve_tool_names(self) -> Optional[List[str]]:
            return ["inspect_options", "test_option_plan"]

        def _get_synthesis_tool_names(self) -> Optional[List[str]]:
            return ["inspect_trajectories", "visualize_state"]

    obj = _Approach()
    assert obj._get_solve_tool_names() == [
        "inspect_options", "test_option_plan"
    ]
    assert obj._get_synthesis_tool_names() == [
        "inspect_trajectories", "visualize_state"
    ]


def test_get_allowed_tool_list_passes_dynamic_names_through() -> None:
    """The allowlist must include dynamic tool names verbatim — the declared
    list is the single source of truth, with no silent filtering against
    ``ALL_TOOL_NAMES``."""
    allowed = get_allowed_tool_list([
        "inspect_options",  # static
        "run_python",  # dynamic synthesis tool
        "evaluate_predicate_quality",  # dynamic predicate-synthesis
    ])
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    assert allowed == [
        f"{prefix}inspect_options",
        f"{prefix}run_python",
        f"{prefix}evaluate_predicate_quality",
    ]
