"""Custom MCP tool definitions for the agent SDK approach."""
import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from predicators.agent_sdk.proposal_parser import ProposalBundle, \
    build_exec_context, exec_code_safely, validate_predicate
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import CausalProcess, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type

MCP_SERVER_NAME = "predicator_tools"

INSPECTION_TOOL_NAMES = [
    "inspect_types",
    "inspect_predicates",
    "inspect_processes",
    "inspect_options",
    "inspect_trajectories",
    "inspect_train_tasks",
    "inspect_planning_results",
    "inspect_iteration_history",
]
PROPOSAL_TOOL_NAMES = [
    "propose_types",
    "propose_predicates",
    "propose_object_augmentor",
    "propose_processes",
    "propose_options",
]
RETRACTION_TOOL_NAMES = [
    "retract_abstractions",
]
TESTING_TOOL_NAMES = [
    "test_predicate_on_states",
    "test_planning",
    "test_option_plan",
]
PLANNING_TOOL_NAMES = [
    "generate_bilevel_plan",
    "generate_abstract_plan",
]
ALL_TOOL_NAMES = (INSPECTION_TOOL_NAMES + PROPOSAL_TOOL_NAMES +
                  RETRACTION_TOOL_NAMES + TESTING_TOOL_NAMES +
                  PLANNING_TOOL_NAMES)


def get_allowed_tool_list(tool_names: Optional[List[str]] = None) -> List[str]:
    """Compute the allowed_tools list for the agent SDK.

    Args:
        tool_names: If provided, only include these tool names.
            If None, include all tools.
    """
    prefix = f"mcp__{MCP_SERVER_NAME}__"
    names = ALL_TOOL_NAMES if tool_names is None else \
        [n for n in tool_names if n in set(ALL_TOOL_NAMES)]
    return [f"{prefix}{n}" for n in names]


@dataclass
class ToolContext:
    """Shared mutable state between the approach and MCP tools."""
    types: Set[Type] = field(default_factory=set)
    predicates: Set[Predicate] = field(default_factory=set)
    processes: Set[CausalProcess] = field(default_factory=set)
    options: Set[ParameterizedOption] = field(default_factory=set)
    train_tasks: List[Task] = field(default_factory=list)
    offline_trajectories: List[LowLevelTrajectory] = field(
        default_factory=list)
    online_trajectories: List[LowLevelTrajectory] = field(default_factory=list)
    example_state: Optional[State] = None
    option_model: Optional[_OptionModelBase] = None
    current_task: Optional[Task] = None
    iteration_proposals: ProposalBundle = field(default_factory=ProposalBundle)
    planning_results: Dict[str, Any] = field(default_factory=dict)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    option_builder_context: Dict[str, Any] = field(default_factory=dict)
    proposals_disabled: bool = False  # set True during test-time solving


def _text_result(text: str) -> Dict[str, Any]:
    """Helper to format a successful text result."""
    return {"content": [{"type": "text", "text": text}]}


def _error_result(text: str) -> Dict[str, Any]:
    """Helper to format an error result."""
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def create_mcp_tools(ctx: ToolContext,
                     tool_names: Optional[List[str]] = None) -> list:
    """Create MCP tools with the given ToolContext via closures.

    Args:
        ctx: Shared mutable state between the approach and MCP tools.
        tool_names: If provided, only return tools with these names.
            If None, return all tools.

    Returns a list of SdkMcpTool objects to pass to create_sdk_mcp_server.
    """
    from claude_agent_sdk import tool

    # ===== INSPECTION TOOLS =====

    @tool("inspect_types", "List all object types and their features", {})
    async def inspect_types(args: Dict[str, Any]) -> Dict[str, Any]:
        lines = []
        for t in sorted(ctx.types, key=lambda t: t.name):
            features = ", ".join(
                t.feature_names) if t.feature_names else "(no features)"
            parent_str = f" (parent: {t.parent.name})" if t.parent else ""
            lines.append(f"- {t.name}{parent_str}: [{features}]")
        if not lines:
            return _text_result("No types defined.")
        return _text_result("Current types:\n" + "\n".join(lines))

    @tool("inspect_predicates",
          "List all predicates and their type signatures", {})
    async def inspect_predicates(args: Dict[str, Any]) -> Dict[str, Any]:
        lines = []
        for p in sorted(ctx.predicates, key=lambda p: p.name):
            type_sig = ", ".join(t.name for t in p.types)
            lines.append(f"- {p.name}({type_sig})")
        if not lines:
            return _text_result("No predicates defined.")
        return _text_result("Current predicates:\n" + "\n".join(lines))

    @tool("inspect_processes",
          "List all processes with conditions, effects, and delays", {})
    async def inspect_processes(args: Dict[str, Any]) -> Dict[str, Any]:
        lines = []
        for proc in sorted(ctx.processes, key=lambda p: p.name):
            conds = ", ".join(str(a) for a in sorted(proc.condition_at_start))
            adds = ", ".join(str(a) for a in sorted(proc.add_effects))
            dels = ", ".join(str(a) for a in sorted(proc.delete_effects))
            lines.append(f"- {proc.name}\n"
                         f"    Conditions: {{{conds}}}\n"
                         f"    Add effects: {{{adds}}}\n"
                         f"    Delete effects: {{{dels}}}\n"
                         f"    Delay: {proc.delay_distribution}")
        if not lines:
            return _text_result("No processes defined.")
        return _text_result("Current processes:\n" + "\n".join(lines))

    @tool("inspect_options",
          "List all options with type signatures and parameter dimensions", {})
    async def inspect_options(args: Dict[str, Any]) -> Dict[str, Any]:
        lines = []
        for opt in sorted(ctx.options, key=lambda o: o.name):
            type_sig = ", ".join(t.name for t in opt.types)
            dim = opt.params_space.shape[0] if opt.params_space.shape else 0
            lines.append(f"- {opt.name}({type_sig}), params_dim={dim}")
        if not lines:
            return _text_result("No options defined.")
        return _text_result("Current options:\n" + "\n".join(lines))

    @tool(
        "inspect_trajectories",
        "Inspect trajectory data. Returns state features and/or atoms.",
        {
            "type": "object",
            "properties": {
                "traj_idx": {
                    "type": "integer",
                    "description": "Trajectory index (0-based)"
                },
                "include_states": {
                    "type": "boolean",
                    "description": "Include state feature dicts",
                    "default": True
                },
                "include_atoms": {
                    "type": "boolean",
                    "description": "Include abstract atoms",
                    "default": False
                },
                "max_timesteps": {
                    "type": "integer",
                    "description": "Max timesteps to show",
                    "default": 10
                },
            },
            "required": ["traj_idx"],
        },
    )
    async def inspect_trajectories(args: Dict[str, Any]) -> Dict[str, Any]:
        from predicators import utils

        traj_idx = args["traj_idx"]
        include_states = args.get("include_states", True)
        include_atoms = args.get("include_atoms", False)
        max_timesteps = args.get("max_timesteps", 10)

        all_trajs = ctx.offline_trajectories + ctx.online_trajectories
        if traj_idx < 0 or traj_idx >= len(all_trajs):
            return _error_result(f"Invalid traj_idx {traj_idx}. "
                                 f"Available: 0-{len(all_trajs)-1}")

        traj = all_trajs[traj_idx]
        lines = [
            f"Trajectory {traj_idx}: {len(traj.states)} states, "
            f"{len(traj.actions)} actions"
        ]

        for t_step, state in enumerate(traj.states[:max_timesteps]):
            lines.append(f"\n--- Timestep {t_step} ---")
            if include_states:
                state_dict = {}
                for obj in sorted(state, key=lambda o: str(o)):
                    obj_feats = {}
                    for feat in obj.type.feature_names:
                        val = state.get(obj, feat)
                        obj_feats[feat] = round(float(val), 4) \
                            if isinstance(val, (float, int)) else str(val)
                    state_dict[str(obj)] = obj_feats
                lines.append(f"State: {json.dumps(state_dict, indent=2)}")
            if include_atoms:
                atoms = utils.abstract(state, ctx.predicates)
                atoms_str = ", ".join(str(a) for a in sorted(atoms))
                lines.append(f"Atoms: {{{atoms_str}}}")
            if t_step < len(traj.actions):
                act = traj.actions[t_step]
                opt = act.get_option()
                lines.append(f"Action: {opt.name}({opt.objects})")

        if len(traj.states) > max_timesteps:
            lines.append(
                f"\n... ({len(traj.states) - max_timesteps} more timesteps)")

        return _text_result("\n".join(lines))

    @tool(
        "inspect_train_tasks",
        "Inspect training tasks (goals, initial atoms, objects)",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Task index (0-based). Omit to see summary of all.",
                },
            },
        },
    )
    async def inspect_train_tasks(args: Dict[str, Any]) -> Dict[str, Any]:
        from predicators import utils

        task_idx = args.get("task_idx")

        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
            goal_str = ", ".join(str(g) for g in sorted(task.goal))
            init_atoms = utils.abstract(task.init, ctx.predicates)
            atoms_str = ", ".join(str(a) for a in sorted(init_atoms))
            objects = sorted(task.init, key=lambda o: str(o))
            obj_str = ", ".join(f"{o.name}:{o.type.name}" for o in objects)
            return _text_result(f"Task {task_idx}:\n"
                                f"  Goal: {{{goal_str}}}\n"
                                f"  Initial atoms: {{{atoms_str}}}\n"
                                f"  Objects: [{obj_str}]")
        else:
            lines = [f"Total tasks: {len(ctx.train_tasks)}"]
            for i, task in enumerate(ctx.train_tasks[:10]):
                goal_str = ", ".join(str(g) for g in sorted(task.goal))
                lines.append(f"  Task {i}: goal={{{goal_str}}}")
            if len(ctx.train_tasks) > 10:
                lines.append(f"  ... ({len(ctx.train_tasks) - 10} more tasks)")
            return _text_result("\n".join(lines))

    @tool("inspect_planning_results",
          "Get latest planning performance metrics", {})
    async def inspect_planning_results(args: Dict[str, Any]) -> Dict[str, Any]:
        if not ctx.planning_results:
            return _text_result("No planning results available yet.")
        return _text_result(
            json.dumps(ctx.planning_results, indent=2, default=str))

    @tool("inspect_iteration_history", "Get summaries of all past iterations",
          {})
    async def inspect_iteration_history(
            args: Dict[str, Any]) -> Dict[str, Any]:
        if not ctx.iteration_history:
            return _text_result("No iteration history available yet.")
        lines = []
        for entry in ctx.iteration_history:
            lines.append(json.dumps(entry, indent=2, default=str))
        return _text_result("\n---\n".join(lines))

    # ===== PROPOSAL TOOLS =====

    @tool(
        "propose_types",
        "Propose new types. Code must define `proposed_types` (a list of "
        "Type objects).",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code defining proposed_types"
                },
                "description": {
                    "type": "string",
                    "description": "Why these types are needed"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_types(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_types:
            return _error_result("Type proposals are disabled.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types, ctx.predicates, ctx.options)
        result, error = exec_code_safely(code, exec_ctx, "proposed_types")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not isinstance(result, (list, set)):
            return _error_result(
                f"proposed_types must be a list/set, got {type(result)}")
        for t in result:
            if not isinstance(t, Type):
                return _error_result(
                    f"Each item must be a Type, got {type(t)}: {t}")
        proposed = set(result)
        ctx.iteration_proposals.proposed_types |= proposed
        names = [t.name for t in proposed]
        logging.info(f"Agent proposed types: {names}")
        return _text_result(
            f"Successfully proposed {len(proposed)} types: {names}")

    @tool(
        "propose_predicates",
        "Propose new predicates. Code must define `proposed_predicates` "
        "(a list of Predicate objects).",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code defining proposed_predicates"
                },
                "description": {
                    "type": "string",
                    "description": "What these predicates capture"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_predicates(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_predicates:
            return _error_result("Predicate proposals are disabled.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types, ctx.predicates, ctx.options)
        result, error = exec_code_safely(code, exec_ctx, "proposed_predicates")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not isinstance(result, (list, set)):
            return _error_result(
                f"proposed_predicates must be a list/set, got {type(result)}")

        validated = []
        errors = []
        for pred in result:
            if not isinstance(pred, Predicate):
                errors.append(f"Not a Predicate: {type(pred)}: {pred}")
                continue
            if ctx.example_state is not None:
                err = validate_predicate(pred, ctx.types, ctx.example_state)
                if err:
                    errors.append(f"{pred.name}: {err}")
                    continue
            validated.append(pred)

        proposed = set(validated)
        ctx.iteration_proposals.proposed_predicates |= proposed
        names = [p.name for p in proposed]
        logging.info(f"Agent proposed predicates: {names}")

        msg = f"Successfully proposed {len(proposed)} predicates: {names}"
        if errors:
            msg += f"\n\nValidation errors ({len(errors)}):\n" + \
                   "\n".join(errors)
        return _text_result(msg)

    @tool(
        "propose_object_augmentor",
        "Propose a task augmentation function. Code must define "
        "`augment_task(task) -> Task`.",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type":
                    "string",
                    "description":
                    "Python code defining augment_task(task) -> Task"
                },
                "description": {
                    "type": "string",
                    "description": "What the augmentor does"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_object_augmentor(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_objects:
            return _error_result("Object augmentor proposals are disabled.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types, ctx.predicates, ctx.options)
        result, error = exec_code_safely(code, exec_ctx, "augment_task")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not callable(result):
            return _error_result(
                f"augment_task must be callable, got {type(result)}")

        # Test on first train task if available
        if ctx.train_tasks:
            try:
                test_task = ctx.train_tasks[0]
                augmented = result(test_task)
                orig_objs = set(test_task.init)
                new_objs = set(augmented.init) - orig_objs
                obj_names = [str(o) for o in sorted(new_objs, key=str)]
            except Exception:
                return _error_result(f"augment_task failed on test task:\n"
                                     f"{traceback.format_exc()}")
        else:
            obj_names = ["(no tasks to test on)"]

        ctx.iteration_proposals.augment_task_fn = result
        ctx.iteration_proposals.augment_task_code = code
        logging.info(f"Agent proposed augmentor adding objects: {obj_names}")
        return _text_result(
            f"Successfully proposed augmentor. Test added objects: {obj_names}"
        )

    @tool(
        "propose_processes",
        "Propose new causal processes. Code must define "
        "`proposed_processes` (a list of CausalProcess objects).",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code defining proposed_processes"
                },
                "description": {
                    "type": "string",
                    "description": "What these processes model"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_processes(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_processes:
            return _error_result("Process proposals are disabled.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types, ctx.predicates, ctx.options)
        result, error = exec_code_safely(code, exec_ctx, "proposed_processes")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not isinstance(result, (list, set)):
            return _error_result(
                f"proposed_processes must be a list/set, got {type(result)}")
        for proc in result:
            if not isinstance(proc, CausalProcess):
                return _error_result(
                    f"Each item must be a CausalProcess, got {type(proc)}")
        proposed = set(result)
        ctx.iteration_proposals.proposed_processes |= proposed
        names = [p.name for p in proposed]
        logging.info(f"Agent proposed processes: {names}")
        return _text_result(
            f"Successfully proposed {len(proposed)} processes: {names}")

    @tool(
        "propose_options",
        "Propose new parameterized options. Code must define "
        "`proposed_options` (a list of ParameterizedOption objects).",
        {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code defining proposed_options"
                },
                "description": {
                    "type": "string",
                    "description": "What these options do"
                },
            },
            "required": ["code", "description"],
        },
    )
    async def propose_options(args: Dict[str, Any]) -> Dict[str, Any]:
        if not CFG.agent_sdk_propose_options:
            return _error_result("Option proposals are disabled.")
        if ctx.proposals_disabled:
            return _error_result(
                "Proposals are disabled during test-time solving. "
                "Options can only be proposed during learning.")
        code = args["code"]
        exec_ctx = build_exec_context(ctx.types, ctx.predicates, ctx.options,
                                      extra_context=ctx.option_builder_context)
        result, error = exec_code_safely(code, exec_ctx, "proposed_options")
        if error:
            return _error_result(f"Code execution failed:\n{error}")
        if not isinstance(result, (list, set)):
            return _error_result(
                f"proposed_options must be a list/set, got {type(result)}")
        for opt in result:
            if not isinstance(opt, ParameterizedOption):
                return _error_result(
                    f"Each item must be a ParameterizedOption, "
                    f"got {type(opt)}")
        proposed = set(result)
        ctx.iteration_proposals.proposed_options |= proposed
        names = [o.name for o in proposed]
        logging.info(f"Agent proposed options: {names}")
        return _text_result(
            f"Successfully proposed {len(proposed)} options: {names}")

    # ===== RETRACTION TOOLS =====

    @tool(
        "retract_abstractions",
        "Remove previously proposed abstractions that are no longer needed. "
        "Specify names of predicates, processes, options, or helper types to "
        "remove, and/or set clear_object_augmentor to remove the augmentor.",
        {
            "type": "object",
            "properties": {
                "predicate_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Names of predicates to remove",
                },
                "process_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Names of processes to remove",
                },
                "option_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Names of options to remove",
                },
                "type_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Names of helper types to remove",
                },
                "clear_object_augmentor": {
                    "type": "boolean",
                    "description":
                    "Set to true to remove the object augmentor",
                },
                "reason": {
                    "type": "string",
                    "description": "Why these abstractions are being removed",
                },
            },
            "required": ["reason"],
        },
    )
    async def retract_abstractions(args: Dict[str, Any]) -> Dict[str, Any]:
        if ctx.proposals_disabled:
            return _error_result(
                "Retractions are disabled during test-time solving. "
                "Abstractions can only be retracted during learning.")
        pred_names = set(args.get("predicate_names") or [])
        proc_names = set(args.get("process_names") or [])
        opt_names = set(args.get("option_names") or [])
        type_names = set(args.get("type_names") or [])
        clear_augmentor = bool(args.get("clear_object_augmentor", False))

        if not any(
            [pred_names, proc_names, opt_names, type_names, clear_augmentor]):
            return _text_result("Nothing to retract.")

        lines = [f"Retracting abstractions. Reason: {args['reason']}"]

        if pred_names:
            existing = {p.name for p in ctx.predicates}
            unknown = pred_names - existing
            valid = pred_names & existing
            ctx.iteration_proposals.retract_predicate_names |= valid
            lines.append(f"Predicates to retract: {sorted(valid)}")
            if unknown:
                lines.append(f"  (unknown, ignored: {sorted(unknown)})")

        if proc_names:
            existing = {p.name for p in ctx.processes}
            unknown = proc_names - existing
            valid = proc_names & existing
            ctx.iteration_proposals.retract_process_names |= valid
            lines.append(f"Processes to retract: {sorted(valid)}")
            if unknown:
                lines.append(f"  (unknown, ignored: {sorted(unknown)})")

        if opt_names:
            existing = {o.name for o in ctx.options}
            unknown = opt_names - existing
            valid = opt_names & existing
            ctx.iteration_proposals.retract_option_names |= valid
            lines.append(f"Options to retract: {sorted(valid)}")
            if unknown:
                lines.append(f"  (unknown, ignored: {sorted(unknown)})")

        if type_names:
            existing = {t.name for t in ctx.types}
            unknown = type_names - existing
            valid = type_names & existing
            ctx.iteration_proposals.retract_type_names |= valid
            lines.append(f"Helper types to retract: {sorted(valid)}")
            if unknown:
                lines.append(f"  (unknown, ignored: {sorted(unknown)})")

        if clear_augmentor:
            ctx.iteration_proposals.retract_object_augmentor = True
            lines.append("Object augmentor will be cleared.")

        logging.info(f"Agent retraction request: {args}")
        return _text_result("\n".join(lines))

    # ===== TESTING TOOLS =====

    @tool(
        "test_predicate_on_states",
        "Test a predicate's truth value across timesteps in a trajectory",
        {
            "type": "object",
            "properties": {
                "predicate_name": {
                    "type": "string",
                    "description": "Name of the predicate to test"
                },
                "traj_idx": {
                    "type": "integer",
                    "description": "Trajectory index"
                },
                "object_names": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Object names to ground the predicate on"
                },
            },
            "required": ["predicate_name", "traj_idx", "object_names"],
        },
    )
    async def test_predicate_on_states(args: Dict[str, Any]) -> Dict[str, Any]:
        pred_name = args["predicate_name"]
        traj_idx = args["traj_idx"]
        object_names = args["object_names"]

        # Find the predicate
        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates
        pred = None
        for p in all_preds:
            if p.name == pred_name:
                pred = p
                break
        if pred is None:
            return _error_result(f"Predicate '{pred_name}' not found.")

        all_trajs = ctx.offline_trajectories + ctx.online_trajectories
        if traj_idx < 0 or traj_idx >= len(all_trajs):
            return _error_result(f"Invalid traj_idx {traj_idx}. "
                                 f"Available: 0-{len(all_trajs)-1}")

        traj = all_trajs[traj_idx]

        # Find objects by name
        objects = []
        for name in object_names:
            found = None
            for obj in traj.states[0]:
                if obj.name == name:
                    found = obj
                    break
            if found is None:
                return _error_result(
                    f"Object '{name}' not found in trajectory {traj_idx}. "
                    f"Available: {[o.name for o in sorted(traj.states[0], key=str)]}"
                )
            objects.append(found)

        results = []
        for t_step, state in enumerate(traj.states):
            try:
                val = pred.holds(state, objects)
                results.append(f"t={t_step}: {val}")
            except Exception as e:
                results.append(f"t={t_step}: ERROR ({e})")

        return _text_result(
            f"Predicate {pred_name}({', '.join(object_names)}) "
            f"over trajectory {traj_idx}:\n" + "\n".join(results))

    @tool(
        "test_planning",
        "Run the task planner on a specific task and report results",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type": "integer",
                    "description": "Task index to plan for"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Planning timeout in seconds",
                    "default": 30
                },
            },
            "required": ["task_idx"],
        },
    )
    async def test_planning(args: Dict[str, Any]) -> Dict[str, Any]:
        from predicators.approaches import ApproachFailure, ApproachTimeout
        from predicators.planning_with_processes import \
            run_task_plan_with_processes_once

        task_idx = args["task_idx"]
        timeout = args.get("timeout", 30)

        if task_idx < 0 or task_idx >= len(ctx.train_tasks):
            return _error_result(f"Invalid task_idx {task_idx}. "
                                 f"Available: 0-{len(ctx.train_tasks)-1}")

        task = ctx.train_tasks[task_idx]
        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates

        try:
            plan, atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                ctx.processes | ctx.iteration_proposals.proposed_processes,
                all_preds,
                ctx.types | ctx.iteration_proposals.proposed_types,
                timeout,
                seed=CFG.seed,
                task_planning_heuristic=CFG.process_task_planning_heuristic,
                max_horizon=float(CFG.horizon))
            plan_desc = " -> ".join(p.name for p in plan)
            return _text_result(
                f"Planning succeeded for task {task_idx}!\n"
                f"Plan length: {len(plan)}\n"
                f"Nodes expanded: {metrics.get('num_nodes_expanded', '?')}\n"
                f"Plan: {plan_desc}")
        except (ApproachFailure, ApproachTimeout, Exception) as e:
            return _text_result(f"Planning failed for task {task_idx}.\n"
                                f"Reason: {type(e).__name__}: {e}")

    @tool(
        "test_option_plan",
        "Execute a sequence of grounded options on a task via the option model "
        "and report the result at each step. Use include_states and/or "
        "include_atoms to control what is shown at each step.",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index to test on. Omit to use "
                    "the current solve-time task (if available)."
                },
                "option_plan": {
                    "type": "array",
                    "description": "Ordered list of options to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option_name": {
                                "type": "string",
                                "description":
                                "Name of the ParameterizedOption"
                            },
                            "object_names": {
                                "type":
                                "array",
                                "items": {
                                    "type": "string"
                                },
                                "description":
                                "Object names to ground the option on"
                            },
                            "params": {
                                "type":
                                "array",
                                "items": {
                                    "type": "number"
                                },
                                "description":
                                "Continuous parameters (empty list if none)"
                            },
                        },
                        "required": ["option_name", "object_names", "params"],
                    },
                },
                "include_states": {
                    "type":
                    "boolean",
                    "description":
                    "Include the full low-level state feature dict after each "
                    "step",
                    "default":
                    False
                },
                "include_atoms": {
                    "type": "boolean",
                    "description":
                    "Include atoms added/deleted after each step",
                    "default": True
                },
            },
            "required": ["option_plan"],
        },
    )
    async def test_option_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        from predicators import utils

        if ctx.option_model is None:
            return _error_result("No option model available in ToolContext.")

        task_idx = args.get("task_idx")
        option_plan_spec = args["option_plan"]
        include_states = args.get("include_states", False)
        include_atoms = args.get("include_atoms", True)

        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
        elif ctx.current_task is not None:
            task = ctx.current_task
            task_idx = "current"
        else:
            return _error_result(
                "No task_idx provided and no current_task set.")
        all_options = ctx.options | ctx.iteration_proposals.proposed_options
        opt_map = {o.name: o for o in all_options}

        state = task.init
        lines = [f"Testing option plan on task {task_idx}:"]

        for step_idx, opt_spec in enumerate(option_plan_spec):
            opt_name = opt_spec["option_name"]
            obj_names = opt_spec["object_names"]
            params = opt_spec["params"]

            if opt_name not in opt_map:
                return _error_result(f"Unknown option '{opt_name}'. "
                                     f"Available: {sorted(opt_map.keys())}")

            param_opt = opt_map[opt_name]

            obj_name_to_obj = {o.name: o for o in state}
            objects = []
            for name in obj_names:
                if name not in obj_name_to_obj:
                    return _error_result(
                        f"Object '{name}' not found in state at step "
                        f"{step_idx}. Available: "
                        f"{sorted(obj_name_to_obj.keys())}")
                objects.append(obj_name_to_obj[name])

            try:
                params_arr = np.array(params, dtype=np.float32)
                option = param_opt.ground(objects, params_arr)
            except Exception as e:
                return _error_result(
                    f"Failed to ground option '{opt_name}' at step "
                    f"{step_idx}: {e}")

            if not option.initiable(state):
                atoms = utils.abstract(state, ctx.predicates)
                atoms_str = ", ".join(str(a) for a in sorted(atoms))
                lines.append(f"Step {step_idx}: {opt_name}({obj_names}) - "
                             f"NOT INITIABLE\n"
                             f"  Current atoms: {{{atoms_str}}}")
                return _text_result("\n".join(lines) +
                                    "\n\nPlan FAILED: option not initiable.")

            try:
                next_state, num_actions = \
                    ctx.option_model.get_next_state_and_num_actions(
                        state, option)
            except Exception as e:
                lines.append(f"Step {step_idx}: {opt_name}({obj_names}) - "
                             f"EXECUTION ERROR: {e}")
                return _text_result("\n".join(lines) +
                                    "\n\nPlan FAILED: execution error.")

            step_line = (f"Step {step_idx}: {opt_name}({obj_names}) "
                         f"({num_actions} actions)")
            if include_atoms:
                atoms_before = utils.abstract(state, ctx.predicates)
                atoms_after = utils.abstract(next_state, ctx.predicates)
                added = atoms_after - atoms_before
                deleted = atoms_before - atoms_after
                step_line += (
                    f"\n  Added:   {{{', '.join(str(a) for a in sorted(added))}}}"
                    f"\n  Deleted: {{{', '.join(str(a) for a in sorted(deleted))}}}"
                )
            if include_states:
                state_dict = {}
                for obj in sorted(next_state, key=lambda o: str(o)):
                    obj_feats = {}
                    for feat in obj.type.feature_names:
                        val = next_state.get(obj, feat)
                        obj_feats[feat] = round(float(val), 4) \
                            if isinstance(val, (float, int)) else str(val)
                    state_dict[str(obj)] = obj_feats
                step_line += f"\n  State: {json.dumps(state_dict, indent=4)}"
            lines.append(step_line)
            state = next_state

        final_atoms = utils.abstract(state, ctx.predicates)
        goal_achieved = task.goal.issubset(final_atoms)
        goal_str = ", ".join(str(g) for g in sorted(task.goal))
        final_atoms_str = ", ".join(str(a) for a in sorted(final_atoms))
        lines.append(f"\nFinal atoms: {{{final_atoms_str}}}")
        lines.append(f"Goal: {{{goal_str}}}")
        lines.append(f"Goal achieved: {goal_achieved}")
        return _text_result("\n".join(lines))

    # ===== PLANNING TOOLS =====

    @tool(
        "generate_bilevel_plan",
        "Generate a concrete option plan using the bilevel planner. Returns "
        "grounded options with sampled continuous parameters, simulated "
        "step-by-step via the option model.",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index. Omit to use the current "
                    "solve-time task (if available)."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Planning timeout in seconds",
                    "default": 30
                },
            },
        },
    )
    async def generate_bilevel_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        from predicators import utils
        from predicators.approaches import ApproachFailure, ApproachTimeout
        from predicators.planning_with_processes import \
            run_task_plan_with_processes_once

        task_idx = args.get("task_idx")
        timeout = args.get("timeout", 30)

        # Resolve task
        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
            task_label = f"train task {task_idx}"
        elif ctx.current_task is not None:
            task = ctx.current_task
            task_label = "current task"
        else:
            return _error_result(
                "No task_idx provided and no current_task set.")

        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates
        all_procs = ctx.processes | ctx.iteration_proposals.proposed_processes
        all_types = ctx.types | ctx.iteration_proposals.proposed_types

        # Get abstract plan
        try:
            plan, atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                all_procs,
                all_preds,
                all_types,
                timeout,
                seed=CFG.seed,
                task_planning_heuristic=CFG.process_task_planning_heuristic,
                max_horizon=float(CFG.horizon))
        except (ApproachFailure, ApproachTimeout, Exception) as e:
            return _text_result(f"Planning failed for {task_label}.\n"
                                f"Reason: {type(e).__name__}: {e}")

        if not plan:
            return _text_result(
                f"Planner returned empty plan for {task_label}.")

        # Sample options and simulate
        rng = np.random.default_rng(CFG.seed)
        state = task.init
        lines = [
            f"Bilevel plan for {task_label} "
            f"({len(plan)} steps, "
            f"{metrics.get('num_nodes_expanded', '?')} nodes expanded):"
        ]

        option_plan_lines = []
        for step_idx, ground_proc in enumerate(plan):
            try:
                option = ground_proc.sample_option(state, task.goal, rng)
            except Exception as e:
                lines.append(
                    f"Step {step_idx}: {ground_proc.name}"
                    f"({', '.join(str(o) for o in ground_proc.objects)}) "
                    f"- SAMPLE FAILED: {e}")
                break

            # Format option
            obj_strs = ", ".join(f"{o.name}:{o.type.name}"
                                 for o in option.objects)
            params_str = ", ".join(f"{p:.4f}" for p in option.params)
            option_line = f"{option.name}({obj_strs})[{params_str}]"
            option_plan_lines.append(option_line)

            # Simulate
            if ctx.option_model is not None:
                try:
                    next_state, num_actions = \
                        ctx.option_model.get_next_state_and_num_actions(
                            state, option)
                    atoms_before = utils.abstract(state, all_preds)
                    atoms_after = utils.abstract(next_state, all_preds)
                    added = atoms_after - atoms_before
                    deleted = atoms_before - atoms_after
                    lines.append(
                        f"Step {step_idx}: {option_line} "
                        f"({num_actions} actions)"
                        f"\n  Added:   "
                        f"{{{', '.join(str(a) for a in sorted(added))}}}"
                        f"\n  Deleted: "
                        f"{{{', '.join(str(a) for a in sorted(deleted))}}}")
                    state = next_state
                except Exception as e:
                    lines.append(f"Step {step_idx}: {option_line} "
                                 f"- SIMULATION ERROR: {e}")
                    break
            else:
                lines.append(f"Step {step_idx}: {option_line}")

        # Check goal
        if ctx.option_model is not None:
            final_atoms = utils.abstract(state, all_preds)
            goal_achieved = task.goal.issubset(final_atoms)
            lines.append(f"\nGoal achieved: {goal_achieved}")

        lines.append(f"\n## Option Plan (copy-paste format):")
        lines.extend(option_plan_lines)

        return _text_result("\n".join(lines))

    @tool(
        "generate_abstract_plan",
        "Generate an abstract plan skeleton without continuous parameters. "
        "Returns option names and objects with parameter space info so you "
        "can fill in continuous parameters yourself.",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Train task index. Omit to use the current "
                    "solve-time task (if available)."
                },
                "timeout": {
                    "type": "integer",
                    "description": "Planning timeout in seconds",
                    "default": 30
                },
            },
        },
    )
    async def generate_abstract_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        from predicators.approaches import ApproachFailure, ApproachTimeout
        from predicators.planning_with_processes import \
            run_task_plan_with_processes_once

        task_idx = args.get("task_idx")
        timeout = args.get("timeout", 30)

        # Resolve task
        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
            task_label = f"train task {task_idx}"
        elif ctx.current_task is not None:
            task = ctx.current_task
            task_label = "current task"
        else:
            return _error_result(
                "No task_idx provided and no current_task set.")

        all_preds = ctx.predicates | ctx.iteration_proposals.proposed_predicates
        all_procs = ctx.processes | ctx.iteration_proposals.proposed_processes
        all_types = ctx.types | ctx.iteration_proposals.proposed_types

        try:
            plan, atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                all_procs,
                all_preds,
                all_types,
                timeout,
                seed=CFG.seed,
                task_planning_heuristic=CFG.process_task_planning_heuristic,
                max_horizon=float(CFG.horizon))
        except (ApproachFailure, ApproachTimeout, Exception) as e:
            return _text_result(f"Planning failed for {task_label}.\n"
                                f"Reason: {type(e).__name__}: {e}")

        if not plan:
            return _text_result(
                f"Planner returned empty plan for {task_label}.")

        lines = [
            f"Abstract plan for {task_label} "
            f"({len(plan)} steps, "
            f"{metrics.get('num_nodes_expanded', '?')} nodes expanded):",
            "",
        ]

        for step_idx, ground_proc in enumerate(plan):
            obj_strs = ", ".join(f"{o.name}:{o.type.name}"
                                 for o in ground_proc.option_objs)
            option = ground_proc.option
            params_dim = option.params_space.shape[0]
            if params_dim > 0:
                low = option.params_space.low.tolist()
                high = option.params_space.high.tolist()
                param_info = (f"  params_dim={params_dim}, "
                              f"low={low}, high={high}")
            else:
                param_info = "  (no continuous params)"
            lines.append(
                f"Step {step_idx}: {option.name}({obj_strs})\n{param_info}")

        # Include conditions for context
        lines.append("\n## Process conditions:")
        for step_idx, ground_proc in enumerate(plan):
            conds = ", ".join(
                str(a) for a in sorted(ground_proc.condition_at_start))
            adds = ", ".join(str(a) for a in sorted(ground_proc.add_effects))
            dels = ", ".join(
                str(a) for a in sorted(ground_proc.delete_effects))
            lines.append(f"Step {step_idx} ({ground_proc.name}):"
                         f"\n  Conditions: {{{conds}}}"
                         f"\n  Add effects: {{{adds}}}"
                         f"\n  Delete effects: {{{dels}}}")

        return _text_result("\n".join(lines))

    _all = {
        "inspect_types": inspect_types,
        "inspect_predicates": inspect_predicates,
        "inspect_processes": inspect_processes,
        "inspect_options": inspect_options,
        "inspect_trajectories": inspect_trajectories,
        "inspect_train_tasks": inspect_train_tasks,
        "inspect_planning_results": inspect_planning_results,
        "inspect_iteration_history": inspect_iteration_history,
        "propose_types": propose_types,
        "propose_predicates": propose_predicates,
        "propose_object_augmentor": propose_object_augmentor,
        "propose_processes": propose_processes,
        "propose_options": propose_options,
        "retract_abstractions": retract_abstractions,
        "test_predicate_on_states": test_predicate_on_states,
        "test_planning": test_planning,
        "test_option_plan": test_option_plan,
        "generate_bilevel_plan": generate_bilevel_plan,
        "generate_abstract_plan": generate_abstract_plan,
    }
    if tool_names is None:
        return list(_all.values())
    return [_all[n] for n in tool_names if n in _all]
