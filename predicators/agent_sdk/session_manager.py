"""Agent session lifecycle management for Claude SDK."""
import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

from predicators.agent_sdk.response_parser import parse_message
from predicators.agent_sdk.sandbox_prompts import truncate
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
        self._conversation_log: List[Dict[str, Any]] = []
        self._current_log_meta: Dict[str, Any] = {}

    @property
    def session_id(self) -> Optional[str]:
        """Return the current session ID."""
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
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient \
            # pylint: disable=import-outside-toplevel

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

    def _init_incremental_log(self, query: str) -> Optional[str]:
        """Initialize log file for incremental writing.

        Returns filepath.
        """
        if not CFG.log_file:
            return None

        self._query_count += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_query_{self._query_count:03d}_{timestamp}.json"
        filepath = os.path.join(self._log_dir, filename)
        os.makedirs(self._log_dir, exist_ok=True)

        self._current_log_meta = {
            "query_number": self._query_count,
            "timestamp": timestamp,
            "query": query,
            "session_id": self._session_id,
        }
        # Write initial state (empty response)
        self._flush_log(filepath, [])
        return filepath

    def _flush_log(self, filepath: str, response: List[Dict[str,
                                                            Any]]) -> None:
        """Rewrite log file with current accumulated response."""
        log_data = {**self._current_log_meta, "response": response}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str)

    async def query(self, message: str) -> List[Dict[str, Any]]:
        """Send a message to the agent and collect all response messages.

        Returns a list of dicts with message content for logging.
        """
        if not self._started:
            await self.start_session()

        collected: List[Dict[str, Any]] = []
        log_path = self._init_incremental_log(message)

        try:
            await self._client.query(message)
            async for msg in self._client.receive_response():
                entry = parse_message(msg)
                if entry is None:
                    continue
                collected.append(entry)

                # Log side-effects
                if entry["type"] == "assistant":
                    for block in entry.get("content", []):
                        if block.get("type") == "text":
                            logging.debug("Agent: %s...", block["text"][:200])
                        elif block.get("type") == "tool_use":
                            params = block.get("input") or {}
                            param_summary = ", ".join(
                                f"{k}={truncate(v)}"
                                for k, v in params.items())
                            logging.debug("Agent tool call: %s(%s)",
                                          block["name"], param_summary)
                elif entry["type"] == "result":
                    cost = entry.get("total_cost_usd")
                    turns = entry.get("num_turns")
                    if cost is not None:
                        self._total_cost_usd += cost
                    if turns is not None:
                        self._total_turns += turns
                    logging.info(
                        "Agent iteration complete. Turns: %s, Cost: $%s", turns
                        or '?', cost or '?')

                # Flush log after each message
                if log_path:
                    self._flush_log(log_path, collected)

        except Exception as e:  # pylint: disable=broad-except
            logging.error("Agent session error: %s", e)
            collected.append({"type": "error", "error": str(e)})
            await self._recover_session(message)

        # Final flush to ensure everything is saved
        if log_path:
            self._flush_log(log_path, collected)
            logging.info("Saved agent query/response to %s", log_path)

        # Track in-memory for conversation replay
        self._conversation_log.append({
            "query": message,
            "response": collected,
        })

        return collected

    @property
    def conversation_log(self) -> List[Dict[str, Any]]:
        """Return the in-memory log of all query/response pairs."""
        return self._conversation_log

    async def _recover_session(self, _last_message: str) -> None:
        """Attempt to recover from a session error."""
        logging.warning("Attempting agent session recovery...")
        try:
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:  # pylint: disable=broad-except
                    pass
            self._started = False
            await self.start_session()
            logging.info("Session recovered successfully.")
        except Exception as e:  # pylint: disable=broad-except
            logging.error("Session recovery failed: %s", e)

    async def close(self) -> None:
        """Close the agent session."""
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception as e:  # pylint: disable=broad-except
                logging.warning("Error closing agent session: %s", e)
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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        logging.info("Saved session info to %s", path)
