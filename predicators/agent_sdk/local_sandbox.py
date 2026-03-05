"""Local-sandboxed agent session manager.

Runs ``ClaudeSDKClient`` in-process with ``cwd`` set to a local sandbox
directory.  The agent gets built-in tools (Bash, Read, Write, Edit, Glob,
Grep) plus custom MCP tools, but PreToolUse hooks restrict built-in file
tools to the sandbox directory.

Unlike ``DockerSessionManager``, no Docker container is used -- the agent
runs directly on the host but is confined to the sandbox via hooks.

Curated reference files are copied into ``sandbox/reference/`` for the
agent to read.  The agent can write and run Python scripts in the sandbox.

Usage
-----
When ``CFG.agent_sdk_use_local_sandbox`` is ``True``, the
``AgentSessionMixin`` creates a ``LocalSandboxSessionManager`` in place
of the normal ``AgentSessionManager``::

    manager = LocalSandboxSessionManager(...)
    responses = await manager.query("Solve this task...")
    await manager.close()
"""
import datetime
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

_MAX_PARAM_LEN = 120


def _truncate(value: Any, max_len: int = _MAX_PARAM_LEN) -> str:
    """Return a short string repr of *value*, truncating if needed."""
    s = repr(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s

from predicators.agent_sdk.docker_sandbox import (
    _SANDBOX_SETTINGS,
    _VALIDATE_SANDBOX_SCRIPT,
    _find_repo_root,
)
from predicators.agent_sdk.tools import ToolContext
from predicators.settings import CFG

logger = logging.getLogger(__name__)

BUILTIN_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]

_LOCAL_CLAUDE_MD = """\
# Predicators Agent Sandbox

## Working Directory
Your working directory is the sandbox. All files you create MUST stay here.
Always use relative paths (e.g., `./my_script.py`).

## Python
The Python interpreter is `python3`. The predicators package is available
for import in your scripts:

    python3 -c "from predicators.structs import State, Type; print('OK')"

You can write and run test scripts in the sandbox:

    python3 my_experiment.py

## Reference Files
Curated source files are available in ./reference/ for you to read.
Read these to understand the APIs before writing code.

## Rules
- Do NOT attempt to read or browse files outside the sandbox directory
- Do NOT modify files in ./reference/ — they are for reading only
- Write all your code, experiments, and tests in the sandbox
- Do NOT inspect predicators source code (e.g. via `inspect.getsource()`,
  `inspect.getfile()`, reading `.py` files from site-packages, or any other
  method). Use the MCP tools and reference files instead.
"""

_LOCAL_SANDBOX_SYSTEM_PROMPT = """

## Sandbox Environment
You are running in a local sandbox environment. You have the following
built-in tools available: Bash, Read, Write, Edit, Glob, Grep.

Your workspace is the current directory. All file operations (Read, Write,
Edit, Glob, Grep) are restricted to this directory.

### Writing and Running Code
You can write Python scripts and execute them with `python3`:
```
python3 my_script.py
```
The predicators package is importable in your scripts:
```python
from predicators.structs import State, Type, Object, Predicate
from predicators.structs import ParameterizedOption, Action
```

### Reference Files
Curated API reference files are in ./reference/.
Read these files to understand the system APIs before writing code.

### Rules
- Do NOT try to read or browse files outside the sandbox directory
- Do NOT modify files in ./reference/
- Do NOT inspect predicators source code via `inspect.getsource()`,
  `inspect.getfile()`, or by reading `.py` files from site-packages.
  Use MCP tools and reference files instead.
"""


class LocalSandboxSessionManager:
    """Runs ClaudeSDKClient locally with cwd set to a sandbox directory.

    Matches the ``AgentSessionManager`` / ``DockerSessionManager``
    interface so that all agent-based approaches work unchanged.
    """

    def __init__(
        self,
        system_prompt: str,
        log_dir: str,
        model_name: str,
        tool_context: ToolContext,
        tool_names: Optional[List[str]] = None,
        extra_reference_files: Optional[Dict[str, str]] = None,
    ) -> None:
        self._system_prompt = system_prompt + _LOCAL_SANDBOX_SYSTEM_PROMPT
        self._log_dir = log_dir
        self._model_name = model_name
        self._tool_context = tool_context
        self._tool_names = tool_names
        self._extra_reference_files = extra_reference_files or {}
        self._repo_root = str(_find_repo_root())

        self._total_cost_usd: float = 0.0
        self._total_turns: int = 0
        self._query_count: int = 0
        self._session_id: Optional[str] = None
        self._conversation_log: List[Dict[str, Any]] = []
        self._sandbox_dir: Optional[str] = None
        self._client: Any = None
        self._started = False

    # -- Properties matching session manager interface --

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    @property
    def tool_names(self) -> List[str]:
        """Return short tool names (without MCP prefix)."""
        from predicators.agent_sdk.tools import MCP_SERVER_NAME
        prefix = f"mcp__{MCP_SERVER_NAME}__"
        names = list(BUILTIN_TOOLS)
        if self._tool_names:
            names += self._tool_names
        return [
            t[len(prefix):] if t.startswith(prefix) else t
            for t in names
        ]

    @property
    def conversation_log(self) -> List[Dict[str, Any]]:
        """Return the in-memory log of all query/response pairs."""
        return self._conversation_log

    # -- Sandbox setup --

    def _ensure_sandbox_dir(self) -> None:
        """Create and populate the sandbox directory if it doesn't exist.

        Sets up:
        - ``reference/`` with curated files from the host repo
        - ``CLAUDE.md`` with agent instructions
        - ``.claude/settings.json`` with PreToolUse hooks
        - ``.claude/validate_sandbox.py`` hook script
        - ``.git/`` marker so Claude CLI treats the sandbox as project root
        """
        if self._sandbox_dir is not None:
            return

        self._sandbox_dir = os.path.abspath(
            os.path.join(self._log_dir, "sandbox"))
        os.makedirs(self._sandbox_dir, exist_ok=True)
        sandbox = Path(self._sandbox_dir)
        logger.info("Setting up local sandbox: %s", self._sandbox_dir)

        # 1. Copy reference files
        registry = dict(self._extra_reference_files)
        ref_dir = sandbox / "reference"
        for dest_rel, src_rel in registry.items():
            src = Path(self._repo_root) / src_rel
            dest = ref_dir / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(str(src), str(dest))
            else:
                logger.warning("Reference file not found: %s", src)

        # 2. .git marker so Claude CLI treats sandbox as project root
        git_marker = sandbox / ".git"
        if not git_marker.exists():
            git_marker.write_text("# marker for Claude CLI project root\n")

        # 3. .claude/settings.json with PreToolUse hooks
        claude_dir = sandbox / ".claude"
        claude_dir.mkdir(exist_ok=True)
        (claude_dir / "settings.json").write_text(
            json.dumps(_SANDBOX_SETTINGS, indent=2) + "\n"
        )

        # 4. validate_sandbox.py hook script
        (claude_dir / "validate_sandbox.py").write_text(
            _VALIDATE_SANDBOX_SCRIPT
        )

        # 5. CLAUDE.md
        (sandbox / "CLAUDE.md").write_text(_LOCAL_CLAUDE_MD)

        logger.info(
            "Local sandbox ready: %d reference files copied",
            sum(1 for d, s in registry.items()
                if (Path(self._repo_root) / s).exists()),
        )

    # -- Session lifecycle --

    async def start_session(self) -> None:
        """Create ClaudeSDKClient with cwd set to the sandbox directory."""
        from claude_agent_sdk import (ClaudeAgentOptions, ClaudeSDKClient,
                                      create_sdk_mcp_server)

        from predicators.agent_sdk.tools import (create_mcp_tools,
                                                 get_allowed_tool_list)

        self._ensure_sandbox_dir()

        # Create MCP tools (closures over tool_context, in-process)
        tools = create_mcp_tools(self._tool_context,
                                 tool_names=self._tool_names)
        mcp_server = create_sdk_mcp_server(
            name="predicator_tools",
            version="1.0.0",
            tools=tools,
        )

        # Built-in tools + custom MCP tools
        mcp_tool_list = get_allowed_tool_list(self._tool_names)
        allowed_tools = BUILTIN_TOOLS + mcp_tool_list

        options = ClaudeAgentOptions(
            allowed_tools=allowed_tools,
            mcp_servers={"predicator_tools": mcp_server},
            permission_mode="bypassPermissions",
            system_prompt=self._system_prompt,
            model=self._model_name,
            max_turns=CFG.agent_sdk_max_agent_turns_per_iteration,
            cwd=self._sandbox_dir,
        )

        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._started = True
        logger.info("Local sandbox session started (cwd=%s)",
                     self._sandbox_dir)

    async def query(self, message: str) -> List[Dict[str, Any]]:
        """Send a message to the agent and collect all response messages."""
        from claude_agent_sdk import (AssistantMessage, ResultMessage,
                                      TextBlock, ToolResultBlock,
                                      ToolUseBlock, UserMessage)

        if not self._started:
            await self.start_session()

        self._query_count += 1
        collected: List[Dict[str, Any]] = []
        log_path = self._init_incremental_log(message)

        try:
            await self._client.query(message)
            async for msg in self._client.receive_response():
                if isinstance(msg, AssistantMessage):
                    entry: Dict[str, Any] = {
                        "type": "assistant", "content": [],
                    }
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            entry["content"].append({
                                "type": "text", "text": block.text,
                            })
                            logging.debug("Agent: %s...", block.text[:200])
                        elif isinstance(block, ToolUseBlock):
                            entry["content"].append({
                                "type": "tool_use",
                                "id": getattr(block, "id", None),
                                "name": block.name,
                                "input": block.input,
                            })
                            params = block.input or {}
                            param_summary = ", ".join(
                                f"{k}={_truncate(v)}"
                                for k, v in params.items())
                            logging.debug(
                                "Agent tool call: %s(%s)",
                                block.name, param_summary)
                        else:
                            block_type = type(block).__name__
                            block_dict: Dict[str, Any] = {"type": block_type}
                            for attr in ("name", "input", "id", "text",
                                         "content", "tool_use_id",
                                         "thinking"):
                                val = getattr(block, attr, None)
                                if val is not None:
                                    block_dict[attr] = val
                            entry["content"].append(block_dict)
                    collected.append(entry)

                elif isinstance(msg, UserMessage):
                    user_entry: Dict[str, Any] = {
                        "type": "user", "content": [],
                    }
                    for block in msg.content:  # type: ignore[union-attr]
                        if isinstance(block, TextBlock):
                            user_entry["content"].append({
                                "type": "text", "text": block.text,
                            })
                        elif isinstance(block, ToolResultBlock):
                            user_entry["content"].append({
                                "type": "tool_result",
                                "tool_use_id": getattr(
                                    block, "tool_use_id", None),
                                "content": getattr(block, "content", None),
                                "is_error": getattr(
                                    block, "is_error", False),
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
                        "total_cost_usd": getattr(
                            msg, "total_cost_usd", None),
                    }
                    collected.append(result_entry)
                    if (hasattr(msg, "total_cost_usd")
                            and msg.total_cost_usd is not None):
                        self._total_cost_usd += msg.total_cost_usd
                    if (hasattr(msg, "num_turns")
                            and msg.num_turns is not None):
                        self._total_turns += msg.num_turns
                    logging.info(
                        "Local sandbox iteration complete. "
                        "Turns: %s, Cost: $%s",
                        getattr(msg, 'num_turns', '?'),
                        getattr(msg, 'total_cost_usd', '?'))

                # Flush log after each message
                if log_path:
                    self._flush_log(log_path, collected)

        except Exception as e:
            logging.error("Local sandbox session error: %s", e)
            collected.append({"type": "error", "error": str(e)})
            await self._recover_session(message)

        # Final flush
        if log_path:
            self._flush_log(log_path, collected)
            logging.info("Saved local sandbox query/response to %s",
                         log_path)

        # Log proposals (matches Docker sandbox logging)
        proposals = self._tool_context.iteration_proposals
        if proposals.proposed_options or proposals.retract_option_names:
            logger.info(
                "Local sandbox proposals: proposed_options=%s, "
                "retract=%s",
                [o.name for o in proposals.proposed_options],
                sorted(proposals.retract_option_names),
            )
            logger.info(
                "After local sandbox query: tool_context.options=%s",
                sorted(o.name for o in self._tool_context.options),
            )

        self._conversation_log.append({
            "query": message,
            "response": collected,
        })

        return collected

    async def close(self) -> None:
        """Close the agent session."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as e:
                logging.warning(
                    "Error closing local sandbox session: %s", e)
            finally:
                self._client = None
                self._started = False

    async def _recover_session(self, last_message: str) -> None:
        """Attempt to recover from a session error."""
        logging.warning("Attempting local sandbox session recovery...")
        try:
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
            self._started = False
            await self.start_session()
            logging.info("Local sandbox session recovered.")
        except Exception as e:
            logging.error("Local sandbox session recovery failed: %s", e)

    def save_session_info(self) -> None:
        """Save session metadata to log directory."""
        os.makedirs(self._log_dir, exist_ok=True)
        info = {
            "session_type": "local_sandbox",
            "session_id": self._session_id,
            "total_cost_usd": self._total_cost_usd,
            "total_turns": self._total_turns,
            "model": self._model_name,
            "sandbox_dir": self._sandbox_dir,
        }
        path = os.path.join(self._log_dir, "session_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        logging.info("Saved session info to %s", path)

    # -- Logging helpers --

    def _init_incremental_log(self, query: str) -> Optional[str]:
        """Initialize log file for incremental writing."""
        if not self._log_dir:
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (f"local_sandbox_query_{self._query_count:03d}_"
                    f"{timestamp}.json")
        filepath = os.path.join(self._log_dir, filename)
        os.makedirs(self._log_dir, exist_ok=True)

        self._current_log_meta = {
            "query_number": self._query_count,
            "timestamp": timestamp,
            "query": query,
            "session_id": self._session_id,
        }
        self._flush_log(filepath, [])
        return filepath

    def _flush_log(self, filepath: str,
                   response: List[Dict[str, Any]]) -> None:
        """Rewrite log file with current accumulated response."""
        log_data = {**self._current_log_meta, "response": response}
        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
