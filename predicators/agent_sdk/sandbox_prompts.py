"""Sandbox-level prompt text shared by the session managers.

``LocalSandboxSessionManager`` and ``DockerSessionManager`` both write
the sandbox ``CLAUDE.md`` (from ``prompts/sandbox_claude_md.md``) and
append the same sandbox suffix to the system prompt. Both describe the
sandbox mechanics only; task guidance lives in the per-phase system
prompts. Sandbox directory scaffolding lives in :mod:`sandbox_setup`.
"""
from predicators.agent_sdk.prompt_templates import render, unwrap_prose_lines
from predicators.agent_sdk.tools import BUILTIN_TOOLS

__all__ = [
    "build_claude_md", "build_sandbox_system_prompt", "unwrap_prose_lines"
]

_BUILTIN_TOOLS_STR = ", ".join(BUILTIN_TOOLS)


def build_claude_md() -> str:
    """The CLAUDE.md content written into every agent sandbox."""
    return render("sandbox_claude_md", "body") + "\n"


def build_sandbox_system_prompt(
    env_description: str = "a local sandbox environment",
    workspace_description: str = "the current directory",
    ref_path: str = "./reference/",
) -> str:
    """Build the system prompt suffix appended for sandbox sessions.

    Args:
        env_description: Short description of the sandbox environment.
        workspace_description: How the workspace directory is described.
        ref_path: Path to reference files shown in examples.
    """
    return ("\n\n## Sandbox Environment\n"
            f"You are running in {env_description}. You have the "
            f"following built-in tools available: {_BUILTIN_TOOLS_STR}."
            "\n\n"
            f"Your workspace is {workspace_description}; all file "
            "operations are restricted to it. The workspace's CLAUDE.md "
            "documents the rest of the layout and rules: the `python3` "
            "interpreter (the predicators package is importable), "
            f"curated API references in {ref_path}, past session logs, "
            "saved scene images and proposed code, and the file-access "
            f"rules. Read the {ref_path} files to understand the system "
            "APIs before writing code.\n")
