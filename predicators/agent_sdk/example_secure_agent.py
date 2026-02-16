"""Example: Secure agent approach with file access capabilities.

This demonstrates how to use the FileAccessValidator to create
a model-free agent that can access files safely.
"""
import os
from typing import Any, List, Optional

from predicators.agent_sdk.secure_file_access import (
    FileAccessValidator,
    create_secure_file_tools,
    create_validator_for_predicators_workspace,
)
from predicators.agent_sdk.session_manager import AgentSessionManager
from predicators.agent_sdk.tools import ToolContext, create_inspection_only_mcp_tools
from predicators.settings import CFG


def create_agent_with_secure_file_access(
    tool_context: ToolContext,
    log_dir: str,
    system_prompt: str,
    allow_file_access: bool = False,
    restrict_to_results: bool = True,
) -> AgentSessionManager:
    """Create an agent session with optional secure file access.

    Args:
        tool_context: ToolContext for domain-specific tools.
        log_dir: Directory for logging.
        system_prompt: System prompt for the agent.
        allow_file_access: Whether to provide file access tools.
        restrict_to_results: If True and allow_file_access is True,
                           only allow access to results/logs directories.

    Returns:
        Configured AgentSessionManager.
    """
    from claude_agent_sdk import create_sdk_mcp_server

    # Start with inspection-only tools
    tools = create_inspection_only_mcp_tools(tool_context)

    # Optionally add secure file access
    tool_prefix = "mcp__predicator_tools__"
    allowed_tools = [
        f"{tool_prefix}inspect_types",
        f"{tool_prefix}inspect_predicates",
        f"{tool_prefix}inspect_options",
        f"{tool_prefix}inspect_trajectories",
        f"{tool_prefix}inspect_train_tasks",
    ]

    if allow_file_access:
        # Get workspace root
        workspace_root = os.getenv("PREDICATORS_ROOT", os.getcwd())
        
        # Create validator with security restrictions
        validator = create_validator_for_predicators_workspace(
            workspace_root=workspace_root,
            restrict_to_results=restrict_to_results,
        )
        
        # Add custom blocked patterns if needed
        # validator.add_blocked_pattern(r"my_secret_file\.txt$")
        
        # Create secure file tools
        file_tools = create_secure_file_tools(validator)
        tools.extend(file_tools)
        
        # Add to allowed tools
        allowed_tools.extend([
            f"{tool_prefix}secure_read_file",
            f"{tool_prefix}secure_list_directory",
            f"{tool_prefix}secure_write_file",
        ])
        
        # Update system prompt to inform agent of restrictions
        system_prompt += (
            "\n\nYou have access to secure file operations. "
            "File access is restricted to specific directories and "
            "sensitive files (like .env, credentials, secrets) are blocked. "
            f"Workspace root: {workspace_root}"
        )

    # Create MCP server with tools
    mcp_server = create_sdk_mcp_server(
        name="predicator_tools",
        version="1.0.0",
        tools=tools,
    )

    # Create session manager
    agent_session = AgentSessionManager(
        system_prompt=system_prompt,
        mcp_server=mcp_server,
        log_dir=log_dir,
        model_name=CFG.agent_open_loop_model_name,
        allowed_tools=allowed_tools,
    )

    return agent_session


# Example usage in an approach:
"""
class SecureModelFreeApproach(BaseApproach):
    def __init__(self, ...):
        super().__init__(...)
        
        self._tool_context = ToolContext(...)
        
        system_prompt = (
            "You are a planning agent. You can inspect task data and "
            "read/write files in restricted directories."
        )
        
        # Create agent with secure file access
        self._agent_session = create_agent_with_secure_file_access(
            tool_context=self._tool_context,
            log_dir="logs/secure_agent",
            system_prompt=system_prompt,
            allow_file_access=True,
            restrict_to_results=True,  # Most secure: only results/logs
        )
"""
