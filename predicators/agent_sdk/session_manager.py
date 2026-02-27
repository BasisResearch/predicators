"""Agent session lifecycle management for Claude SDK."""
import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

from predicators.settings import CFG


class AgentSessionManager:
    """Wraps ClaudeSDKClient for persistent sessions with custom MCP tools."""

    def __init__(self,
                 system_prompt: str,
                 mcp_server: Any,
                 log_dir: str,
                 model_name: str,
                 allowed_tools: Optional[List[str]] = None) -> None:
        self._system_prompt = system_prompt
        self._mcp_server = mcp_server
        self._log_dir = log_dir
        self._model_name = model_name
        self._allowed_tools = allowed_tools
        self._client: Any = None
        self._session_id: Optional[str] = None
        self._total_cost_usd: float = 0.0
        self._total_turns: int = 0
        self._started = False
        self._query_count: int = 0

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        self._session_id = value

    @property
    def tool_names(self) -> List[str]:
        """Return short tool names (without MCP prefix)."""
        if not self._allowed_tools:
            return []
        prefix = "mcp__predicator_tools__"
        return [
            t[len(prefix):] if t.startswith(prefix) else t
            for t in self._allowed_tools
        ]

    async def start_session(self) -> None:
        """Start a new Claude SDK client session."""
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        options = ClaudeAgentOptions(
            allowed_tools=self._allowed_tools or [],
            mcp_servers={"predicator_tools": self._mcp_server},
            permission_mode="bypassPermissions",
            system_prompt=self._system_prompt,
            model=self._model_name,
            max_turns=CFG.agent_sdk_max_agent_turns_per_iteration,
        )

        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        self._started = True
        logging.info("Agent SDK session started.")

    def _save_query_response_log(self, query: str,
                                 response: List[Dict[str, Any]]) -> None:
        """Save query and response to a timestamped JSON file."""
        if not CFG.log_file:
            return

        self._query_count += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_query_{self._query_count:03d}_{timestamp}.json"
        filepath = os.path.join(self._log_dir, filename)

        log_data = {
            "query_number": self._query_count,
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "session_id": self._session_id,
        }

        os.makedirs(self._log_dir, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        logging.info(f"Saved agent query/response to {filepath}")

    async def query(self, message: str) -> List[Dict[str, Any]]:
        """Send a message to the agent and collect all response messages.

        Returns a list of dicts with message content for logging.
        """
        from claude_agent_sdk import AssistantMessage, ResultMessage, \
            TextBlock, ToolResultBlock, ToolUseBlock, UserMessage

        if not self._started:
            await self.start_session()

        collected: List[Dict[str, Any]] = []

        try:
            await self._client.query(message)
            async for msg in self._client.receive_response():
                if isinstance(msg, AssistantMessage):
                    entry: Dict[str, Any] = {
                        "type": "assistant",
                        "content": []
                    }
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            entry["content"].append({
                                "type": "text",
                                "text": block.text
                            })
                            logging.debug(f"Agent: {block.text[:200]}...")
                        elif isinstance(block, ToolUseBlock):
                            entry["content"].append({
                                "type":
                                "tool_use",
                                "id":
                                getattr(block, "id", None),
                                "name":
                                block.name,
                                "input":
                                block.input,
                            })
                            logging.debug(f"Agent tool call: {block.name}")
                    collected.append(entry)
                elif isinstance(msg, UserMessage):
                    user_entry: Dict[str, Any] = {"type": "user", "content": []}
                    for block in msg.content:  # type: ignore[assignment]
                        if isinstance(block, TextBlock):
                            user_entry["content"].append({
                                "type": "text",
                                "text": block.text
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
                            logging.debug(
                                f"Tool result: {getattr(block, 'tool_use_id', '?')}"
                            )
                    collected.append(user_entry)
                elif isinstance(msg, ResultMessage):
                    result_entry = {
                        "type": "result",
                        "num_turns": getattr(msg, "num_turns", None),
                        "total_cost_usd": getattr(msg, "total_cost_usd", None),
                    }
                    collected.append(result_entry)
                    if hasattr(msg, "total_cost_usd") and \
                            msg.total_cost_usd is not None:
                        self._total_cost_usd += msg.total_cost_usd
                    if hasattr(msg, "num_turns") and \
                            msg.num_turns is not None:
                        self._total_turns += msg.num_turns
                    logging.info(
                        f"Agent iteration complete. "
                        f"Turns: {getattr(msg, 'num_turns', '?')}, "
                        f"Cost: ${getattr(msg, 'total_cost_usd', '?')}")
        except Exception as e:
            logging.error(f"Agent session error: {e}")
            collected.append({"type": "error", "error": str(e)})
            # Attempt recovery
            await self._recover_session(message)

        # Save the query and response to a log file
        self._save_query_response_log(message, collected)

        return collected

    async def _recover_session(self, last_message: str) -> None:
        """Attempt to recover from a session error."""
        logging.warning("Attempting agent session recovery...")
        try:
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
            self._started = False
            await self.start_session()
            logging.info("Session recovered successfully.")
        except Exception as e:
            logging.error(f"Session recovery failed: {e}")

    async def close(self) -> None:
        """Close the agent session."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as e:
                logging.warning(f"Error closing agent session: {e}")
            finally:
                self._client = None
                self._started = False

    def save_session_info(self) -> None:
        """Save session metadata to log directory."""
        os.makedirs(self._log_dir, exist_ok=True)
        info = {
            "session_id": self._session_id,
            "total_cost_usd": self._total_cost_usd,
            "total_turns": self._total_turns,
            "model": self._model_name,
        }
        path = os.path.join(self._log_dir, "session_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        logging.info(f"Saved session info to {path}")
