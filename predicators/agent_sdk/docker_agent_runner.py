"""Agent runner for Docker sandbox.

Executed inside the Docker container by DockerSessionManager.  Loads a
pickled ``QueryInput`` dict, creates a ``ClaudeSDKClient`` session with
both Claude built-in tools (Bash, Read, Write, Edit, Glob, Grep, Task*)
and custom predicator MCP tools, queries the agent, and pickles results
back to a shared directory.

The predicators source tree is mounted read-only at ``/opt/predicators``
(via ``PYTHONPATH``) for imports.  Curated reference files are available
at ``/sandbox/reference/``.  A writable sandbox is at ``/sandbox``.
PreToolUse hooks restrict the agent's built-in tools to ``/sandbox/``.

Usage (inside Docker)::

    PYTHONPATH=/opt/predicators python3 \
        /opt/predicators/predicators/agent_sdk/docker_agent_runner.py \
        /data/query_input.pkl /data/query_output.pkl
"""
import asyncio
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional

import dill as pkl

# Bootstrap: import predicators.utils before anything else so that Python
# resolves the circular import chain (structs → utils → image_patch_wrapper
# → structs) in the correct order.  Without this, importing predicators.structs
# first causes image_patch_wrapper to try "from predicators.structs import Mask"
# while structs is still being initialized, raising an ImportError.
import predicators.utils  # noqa: F401, E402  # pylint: disable=unused-import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# pylint: disable=wrong-import-position
from predicators.agent_sdk.log_formatter import \
    format_conversation_markdown  # noqa: E402
from predicators.agent_sdk.response_parser import parse_message  # noqa: E402
from predicators.agent_sdk.sandbox_prompts import truncate  # noqa: E402
from predicators.agent_sdk.tools import BUILTIN_TOOLS  # noqa: E402


async def _run_query(query_input: Dict[str, Any]) -> Dict[str, Any]:
    """Create a ClaudeSDKClient, query the agent, and collect responses."""
    # pylint: disable=import-outside-toplevel
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, \
        create_sdk_mcp_server

    from predicators.agent_sdk.tools import create_mcp_tools, \
        get_allowed_tool_list

    # pylint: enable=import-outside-toplevel

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

    # Incremental log file path (on shared /data volume)
    log_path = query_input.get("log_path")
    log_meta = {"query": query_input.get("message", "")}

    def _flush_log() -> None:
        """Write current conversation state as markdown to the log file."""
        if not log_path:
            return
        try:
            content = format_conversation_markdown(collected,
                                                   title="Docker Query",
                                                   meta=log_meta)
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(content)
        except Exception:  # pylint: disable=broad-except
            pass  # Don't let logging errors break the agent

    try:
        await client.query(query_input["message"])
        async for msg in client.receive_response():
            entry = parse_message(msg)
            if entry is None:
                continue
            collected.append(entry)

            # Docker-specific stderr logging for real-time visibility
            if entry["type"] == "assistant":
                for block in entry.get("content", []):
                    btype = block.get("type", "")
                    if btype == "text":
                        print(f"Agent: {block['text'][:200]}...",
                              file=sys.stderr,
                              flush=True)
                    elif btype == "tool_use":
                        params = block.get("input") or {}
                        param_summary = ", ".join(f"{k}={truncate(v)}"
                                                  for k, v in params.items())
                        print(
                            f"Tool call: {block['name']}"
                            f"({param_summary})",
                            file=sys.stderr,
                            flush=True)
                    elif btype == "ThinkingBlock":
                        thinking = block.get("thinking", "")
                        if thinking:
                            print(f"Thinking: {thinking[:200]}...",
                                  file=sys.stderr,
                                  flush=True)
            elif entry["type"] == "result":
                print(
                    f"Agent iteration complete. "
                    f"Turns: {entry.get('num_turns', '?')}, "
                    f"Cost: ${entry.get('total_cost_usd', '?')}",
                    file=sys.stderr,
                    flush=True)

            # Flush log after each message
            _flush_log()

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Agent session error: %s", e)
        collected.append({"type": "error", "error": str(e)})
        _flush_log()
    finally:
        try:
            await client.disconnect()
        except Exception:  # pylint: disable=broad-except
            pass

    return {
        "responses": collected,
        "iteration_proposals": ctx.iteration_proposals,
    }


def _rehash_objects_after_unpickle(ctx: Any) -> None:
    """Fix stale Object hash caches after cross-process unpickling.

    ``Object.__hash__`` returns a ``cached_property`` (``_hash``) that
    stores ``hash(str(self))``.  Python randomises string hashes across
    processes (PYTHONHASHSEED), so cached values from the *pickling*
    process are stale here.  When the option-model simulator later
    creates fresh Objects (e.g. ``self._robot`` in ``_get_state``),
    their hashes differ from the unpickled Objects, causing KeyError on
    ``State.data`` dict lookups.

    Fix: clear every Object's cached ``_hash`` (and ``_str``) so it is
    re-computed with the current process's hash seed, then rebuild every
    ``State.data`` dict so its internal hash-table is consistent.
    """
    from predicators.structs import \
        State  # pylint: disable=import-outside-toplevel

    seen: set = set()

    def _clear(obj: Any) -> None:
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        obj.__dict__.pop("_hash", None)
        obj.__dict__.pop("_str", None)

    def _process_state(state: Any) -> None:
        if state is None or not isinstance(state, State):
            return
        for obj in list(state.data.keys()):
            _clear(obj)
        # Rebuild dict so Python re-hashes keys with current seed.
        state.data = dict(state.data)

    def _process_atoms(atoms: Any) -> None:
        for atom in atoms:
            for obj in atom.objects:
                _clear(obj)

    def _process_task(task: Any) -> None:
        # Task has .init (State) and .goal (Set[GroundAtom])
        # EnvironmentTask has .init_obs and .goal_description
        if hasattr(task, "init"):
            _process_state(task.init)
        if hasattr(task, "init_obs"):
            _process_state(task.init_obs)
        for attr in ("goal", "alt_goal", "goal_description", "alt_goal_desc"):
            atoms = getattr(task, attr, None)
            if atoms:
                _process_atoms(atoms)

    # Train tasks
    for task in getattr(ctx, "train_tasks", []):
        _process_task(task)

    # Current task
    if ctx.current_task is not None:
        _process_task(ctx.current_task)

    # Example state
    _process_state(getattr(ctx, "example_state", None))

    # Trajectories
    for traj in (getattr(ctx, "offline_trajectories", []) +
                 getattr(ctx, "online_trajectories", [])):
        for state in traj.states:
            _process_state(state)


def main() -> None:
    """Entry point for Docker agent runner."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pkl> <output.pkl>",
              file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    logger.info("Docker agent runner starting: input=%s output=%s", input_path,
                output_path)

    # Load query input
    with open(input_path, "rb") as f:
        query_input = pkl.load(f)

    # Restore host CFG settings (arg-specific settings like
    # max_num_steps_option_rollout are not set by default import)
    if "cfg_snapshot" in query_input:
        from predicators.settings import \
            CFG  # pylint: disable=import-outside-toplevel
        for k, v in query_input["cfg_snapshot"].items():
            setattr(CFG, k, v)

    # Fix stale Object hash caches from cross-process pickling.
    ctx = query_input.get("tool_context")
    if ctx is not None:
        _rehash_objects_after_unpickle(ctx)

    # Recreate option model — the simulator (e.g. PyBullet physics
    # server) is process-local and cannot survive pickling.
    if ctx is not None and ctx.option_model is not None:
        from predicators.option_model import \
            create_option_model  # pylint: disable=import-outside-toplevel
        from predicators.settings import \
            CFG as _cfg  # pylint: disable=import-outside-toplevel
        logger.info("Recreating option model (%s) inside Docker...",
                    _cfg.option_model_name)
        ctx.option_model = create_option_model(_cfg.option_model_name)
        # Sync with all options in context (GT + any previously proposed)
        # after the model has its physics server set up.
        ctx.option_model._name_to_parameterized_option = {  # pylint: disable=protected-access
            o.name: o
            for o in ctx.options
        }

    # Recreate SkillConfig in skill_factory_context — the robot's
    # physics_client_id is process-local and stale after pickling.
    if (ctx is not None
            and ctx.skill_factory_context.get("skill_config") is not None):
        from predicators.settings import \
            CFG as _cfg  # pylint: disable=import-outside-toplevel
        if _cfg.env.startswith("pybullet"):
            try:
                # pylint: disable=import-outside-toplevel,reimported
                from predicators import utils as _utils
                from predicators.envs.base_env import BaseEnv
                from predicators.envs.pybullet_env import PyBulletEnv
                from predicators.ground_truth_models.skill_factories import \
                    SkillConfig

                # Find the PyBulletEnv subclass (envs already imported above
                # by create_option_model → create_new_env).
                env_cls = None
                for cls in _utils.get_all_subclasses(BaseEnv):
                    if (not cls.__abstractmethods__
                            and issubclass(cls, PyBulletEnv)
                            and cls.get_name() == _cfg.env):
                        env_cls = cls
                        break

                if env_cls is None:
                    logger.warning(
                        "Could not find PyBulletEnv for %s; "
                        "skill_config NOT recreated", _cfg.env)
                else:
                    _, robot, _ = env_cls.initialize_pybullet(using_gui=False)
                    ctx.skill_factory_context["skill_config"] = SkillConfig(
                        robot=robot,
                        open_fingers_joint=robot.open_fingers,
                        closed_fingers_joint=robot.closed_fingers,
                        fingers_state_to_joint=(
                            env_cls._fingers_state_to_joint),  # pylint: disable=protected-access
                        max_vel_norm=_cfg.pybullet_max_vel_norm,
                        ik_validate=_cfg.pybullet_ik_validate,
                        robot_init_tilt=getattr(env_cls, 'robot_init_tilt',
                                                0.0),
                        robot_init_wrist=getattr(env_cls, 'robot_init_wrist',
                                                 0.0),
                    )
                    logger.info(
                        "Recreated SkillConfig inside Docker for %s "
                        "(physics_client_id=%d)", _cfg.env,
                        robot.physics_client_id)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Failed to recreate SkillConfig in Docker: %s",
                             e,
                             exc_info=True)

    logger.info("Loaded query input: message length=%d, model=%s",
                len(query_input.get("message", "")),
                query_input.get("model_name", "?"))

    # Run the query
    try:
        query_output = asyncio.run(_run_query(query_input))
    except Exception as e:  # pylint: disable=broad-except
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
