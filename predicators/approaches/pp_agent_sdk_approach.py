"""Agent SDK online process planning approach.

Uses a persistent Claude Agent SDK session to iteratively propose
abstractions (types, predicates, helper objects, processes, options) based
on observed trajectory data and planning feedback.
"""
import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import dill as pkl
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.proposal_parser import ProposalBundle, \
    build_exec_context, exec_code_safely
from predicators.agent_sdk.session_manager import AgentSessionManager
from predicators.agent_sdk.system_prompt import build_iteration_message, \
    build_system_prompt
from predicators.agent_sdk.tools import ToolContext, create_mcp_tools
from predicators.approaches.pp_online_process_learning_approach import \
    OnlineProcessLearningAndPlanningApproach
from predicators.approaches.pp_predicate_invention_approach import \
    PredicateInventionProcessPlanningApproach
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import Action, CausalProcess, Dataset, \
    EndogenousProcess, InteractionResult, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type


class AgentSDKOnlineProcessPlanningApproach(
        PredicateInventionProcessPlanningApproach,
        OnlineProcessLearningAndPlanningApproach):
    """Online process planning approach using Claude Agent SDK.

    Maintains a persistent Claude agent session that iteratively refines
    abstraction proposals based on observed trajectory data and planning
    feedback. The agent cannot see environment source code -- it observes
    the world only through custom MCP tools.
    """

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 option_model: Optional[_OptionModelBase] = None) -> None:
        # Agent-specific attributes (before super().__init__)
        self._helper_types: Set[Type] = set()
        self._augment_task_fn: Optional[Callable] = None
        self._augment_task_code: str = ""
        self._agent_proposed_options: Set[ParameterizedOption] = set()
        self._agent_proposed_processes: Set[CausalProcess] = set()
        self._iteration_history: List[Dict[str, Any]] = []
        self._planning_results: Dict[str, Any] = {}

        # Create ToolContext (shared state between approach and MCP tools)
        self._tool_context = ToolContext(
            types=types,
            predicates=initial_predicates,
            options=initial_options,
            train_tasks=train_tasks,
        )

        # Agent session manager (lazy -- starts on first query)
        self._agent_session: Optional[AgentSessionManager] = None
        self._agent_session_id: Optional[str] = None

        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)

    @classmethod
    def get_name(cls) -> str:
        return "agent_sdk_online_process_planning"

    def _get_log_dir(self) -> str:
        """Get the log directory for agent SDK files."""
        base = CFG.log_file if hasattr(CFG, 'log_file') and CFG.log_file \
            else "logs"
        return os.path.join(base, "agent_sdk")

    def _ensure_agent_session(self) -> None:
        """Create the agent session manager if it doesn't exist yet."""
        if self._agent_session is not None:
            return

        from claude_agent_sdk import create_sdk_mcp_server

        system_prompt = build_system_prompt()
        tools = create_mcp_tools(self._tool_context)
        mcp_server = create_sdk_mcp_server(
            name="predicator_tools",
            version="1.0.0",
            tools=tools,
        )
        log_dir = self._get_log_dir()
        self._agent_session = AgentSessionManager(
            system_prompt=system_prompt,
            mcp_server=mcp_server,
            log_dir=log_dir,
            model_name=CFG.agent_sdk_model_name,
        )
        if self._agent_session_id is not None:
            self._agent_session.session_id = self._agent_session_id

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Store the offline dataset. Do NOT start agent session yet."""
        self._offline_dataset = dataset
        self._tool_context.offline_trajectories = dataset.trajectories
        # Set example state from first trajectory
        if dataset.trajectories:
            self._tool_context.example_state = \
                dataset.trajectories[0].states[0]
        self.save()

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        """Learn from interaction results via the Claude agent."""
        # 1. Convert results to trajectories, append to online dataset
        assert self._requests_train_task_idxs is not None
        for i, result in enumerate(results):
            task_idx = self._requests_train_task_idxs[i]
            traj = LowLevelTrajectory(result.states,
                                      result.actions,
                                      _train_task_idx=task_idx)
            self._online_dataset.append(traj)

        all_trajs = self._offline_dataset.trajectories + \
            self._online_dataset.trajectories

        # 2. Update tool context with current state
        self._sync_tool_context(all_trajs)

        # 3. Run agent iteration
        self._run_agent_iteration(all_trajs)

        # 4. Integrate proposals from tool context
        proposals = self._tool_context.iteration_proposals
        self._integrate_proposals(proposals)

        # 5. Use agent-proposed processes (not data-driven learning)
        # The processes are already integrated in _integrate_proposals
        # Optionally learn parameters for the agent-proposed processes
        if CFG.learn_process_parameters and self._get_current_processes():
            self._learn_process_parameters(all_trajs)

        # 7. Log iteration summary
        summary = self._build_iteration_summary(proposals)
        self._iteration_history.append(summary)
        self._tool_context.iteration_history = self._iteration_history
        logging.info(f"Iteration {self._online_learning_cycle} summary: "
                     f"{json.dumps(summary, default=str)}")

        # 8. Save and log agent responses
        self._save_iteration_logs(self._online_learning_cycle)
        self.save(self._online_learning_cycle)

        # 9. Increment cycle
        self._online_learning_cycle += 1

    def _sync_tool_context(self,
                           all_trajs: List[LowLevelTrajectory]) -> None:
        """Synchronize ToolContext with current approach state."""
        self._tool_context.types = self._types
        self._tool_context.predicates = self._get_current_predicates()
        self._tool_context.processes = self._get_current_processes()
        self._tool_context.options = self._initial_options | \
            self._agent_proposed_options
        self._tool_context.train_tasks = self._train_tasks
        self._tool_context.offline_trajectories = \
            self._offline_dataset.trajectories
        self._tool_context.online_trajectories = \
            self._online_dataset.trajectories
        self._tool_context.planning_results = self._planning_results
        self._tool_context.iteration_history = self._iteration_history

        if all_trajs:
            self._tool_context.example_state = all_trajs[0].states[0]

        # Reset proposals for this iteration
        self._tool_context.iteration_proposals = ProposalBundle()

    def _run_agent_iteration(self,
                             all_trajs: List[LowLevelTrajectory]) -> None:
        """Bridge sync/async and query the Claude agent."""
        self._ensure_agent_session()

        # Build the iteration message
        num_new = len(self._online_dataset.trajectories)
        num_total = len(all_trajs)
        task_success = self._compute_task_success_rate(all_trajs)

        type_str = ", ".join(
            f"{t.name}[{','.join(t.feature_names)}]"
            for t in sorted(self._types, key=lambda t: t.name))
        preds = self._get_current_predicates()
        pred_str = ", ".join(
            f"{p.name}({','.join(t.name for t in p.types)})"
            for p in sorted(preds, key=lambda p: p.name))
        procs = self._get_current_processes()
        proc_str = ", ".join(p.name for p in sorted(procs, key=lambda p: p.name))
        opt_str = ", ".join(o.name for o in sorted(
            self._initial_options, key=lambda o: o.name))

        plan_success = self._planning_results.get("success_str",
                                                  "Not yet evaluated")
        avg_nodes = str(self._planning_results.get("avg_nodes_expanded",
                                                   "N/A"))
        failures = self._planning_results.get("failure_summaries",
                                              "None recorded")

        prev_outcomes = "No previous iterations." if not \
            self._iteration_history else json.dumps(
                self._iteration_history[-1], default=str, indent=2)

        message = build_iteration_message(
            cycle=self._online_learning_cycle,
            num_new_trajs=num_new,
            num_total_trajs=num_total,
            task_success_rate=task_success,
            type_names_with_features=type_str,
            predicate_signatures=pred_str,
            num_predicates=len(preds),
            process_summaries=proc_str,
            num_processes=len(procs),
            option_names=opt_str,
            num_options=len(self._initial_options),
            planning_success=plan_success,
            avg_nodes=avg_nodes,
            failure_summaries=failures,
            previous_iteration_outcomes=prev_outcomes,
        )

        # Save the context message
        self._last_context_message = message

        # Run async query
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                self._last_agent_responses = loop.run_until_complete(
                    self._async_query(message))
            else:
                self._last_agent_responses = loop.run_until_complete(
                    self._async_query(message))
        except RuntimeError:
            self._last_agent_responses = asyncio.run(
                self._async_query(message))

    async def _async_query(self, message: str) -> List[Dict[str, Any]]:
        """Async wrapper for agent query."""
        assert self._agent_session is not None
        return await self._agent_session.query(message)

    def _integrate_proposals(self, proposals: ProposalBundle) -> None:
        """Integrate validated proposals into approach state."""
        # Types
        if proposals.proposed_types:
            self._types = self._types | proposals.proposed_types
            self._helper_types |= proposals.proposed_types
            logging.info(f"Integrated {len(proposals.proposed_types)} "
                         f"new types")

        # Predicates
        if proposals.proposed_predicates:
            self._learned_predicates |= proposals.proposed_predicates
            logging.info(f"Integrated {len(proposals.proposed_predicates)} "
                         f"new predicates")

        # Task augmentor
        if proposals.augment_task_fn is not None:
            self._augment_task_fn = proposals.augment_task_fn
            self._augment_task_code = proposals.augment_task_code or ""
            logging.info("Integrated new task augmentor")

        # Processes (agent-proposed, not data-driven)
        if proposals.proposed_processes:
            self._agent_proposed_processes |= proposals.proposed_processes
            # Also update self._processes to reflect that we're using agent
            # proposals instead of learned processes
            # Cast to the parent's expected type (Set[CausalProcess])
            self._processes = set(self._agent_proposed_processes)  # type: ignore
            logging.info(f"Integrated {len(proposals.proposed_processes)} "
                         f"new processes (total: {len(self._processes)})")

        # Options
        if proposals.proposed_options:
            self._agent_proposed_options |= proposals.proposed_options
            logging.info(f"Integrated {len(proposals.proposed_options)} "
                         f"new options")

    def _get_current_processes(self) -> Set[CausalProcess]:
        """Get current processes including agent-proposed ones."""
        return self._processes | self._agent_proposed_processes

    def _compute_task_success_rate(
            self, trajs: List[LowLevelTrajectory]) -> float:
        """Compute fraction of trajectories that achieved their task goal."""
        if not trajs:
            return 0.0
        successes = 0
        counted = 0
        for traj in trajs:
            if traj._train_task_idx is not None and \
                    traj._train_task_idx < len(self._train_tasks):
                task = self._train_tasks[traj._train_task_idx]
                goal_preds = {a.predicate for a in task.goal}
                final_atoms = utils.abstract(traj.states[-1], goal_preds)
                if task.goal.issubset(final_atoms):
                    successes += 1
                counted += 1
        return successes / max(counted, 1)

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Solve by augmenting task with agent-proposed helper objects."""
        if self._augment_task_fn is not None:
            try:
                task = self._augment_task_fn(task)
            except Exception as e:
                logging.warning(f"Task augmentation failed: {e}. "
                                f"Using original task.")
        return super()._solve(task, timeout)

    def _build_iteration_summary(self,
                                 proposals: ProposalBundle) -> Dict[str, Any]:
        """Build a summary dict of what happened this iteration."""
        return {
            "cycle": self._online_learning_cycle,
            "proposed_types": [t.name for t in proposals.proposed_types],
            "proposed_predicates": [
                p.name for p in proposals.proposed_predicates
            ],
            "proposed_augmentor": proposals.augment_task_code is not None,
            "proposed_processes": [
                p.name for p in proposals.proposed_processes
            ],
            "proposed_options": [
                o.name for o in proposals.proposed_options
            ],
            "errors": proposals.errors,
            "total_predicates": len(self._get_current_predicates()),
            "total_processes": len(self._get_current_processes()),
        }

    def _save_iteration_logs(self, cycle: int) -> None:
        """Save iteration-specific logs to disk."""
        log_dir = os.path.join(self._get_log_dir(), f"iteration_{cycle}")
        os.makedirs(log_dir, exist_ok=True)

        # Context message
        if hasattr(self, '_last_context_message'):
            with open(os.path.join(log_dir, "context_message.txt"), "w") as f:
                f.write(self._last_context_message)

        # Agent responses
        if CFG.agent_sdk_log_agent_responses and \
                hasattr(self, '_last_agent_responses'):
            resp_path = os.path.join(log_dir, "agent_responses.jsonl")
            with open(resp_path, "w") as f:
                for resp in self._last_agent_responses:
                    f.write(json.dumps(resp, default=str) + "\n")

        # Proposals directory
        proposals_dir = os.path.join(log_dir, "proposals")
        os.makedirs(proposals_dir, exist_ok=True)

        proposals = self._tool_context.iteration_proposals
        if proposals.proposed_types:
            with open(os.path.join(proposals_dir, "types.json"), "w") as f:
                json.dump([t.name for t in proposals.proposed_types], f,
                          indent=2)
        if proposals.proposed_predicates:
            with open(os.path.join(proposals_dir,
                                   "predicates_validated.json"), "w") as f:
                json.dump([p.name for p in proposals.proposed_predicates], f,
                          indent=2)
        if proposals.augment_task_code:
            with open(os.path.join(proposals_dir,
                                   "augmentor_code.py"), "w") as f:
                f.write(proposals.augment_task_code)
        if proposals.proposed_processes:
            with open(os.path.join(proposals_dir,
                                   "processes_code.json"), "w") as f:
                json.dump([p.name for p in proposals.proposed_processes], f,
                          indent=2)

        # Session info
        if self._agent_session is not None:
            self._agent_session.save_session_info()

    def save(self, online_learning_cycle: Optional[int] = None) -> None:
        """Save approach state."""
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.AgentSDK", "wb") as f:
            save_dict = {
                "processes": self._processes,
                "learned_predicates": self._learned_predicates,
                "offline_dataset": self._offline_dataset,
                "online_dataset": self._online_dataset,
                "online_learning_cycle": self._online_learning_cycle,
                "helper_types": self._helper_types,
                "augment_task_code": self._augment_task_code,
                "agent_proposed_options": self._agent_proposed_options,
                "agent_proposed_processes": self._agent_proposed_processes,
                "iteration_history": self._iteration_history,
                "agent_session_id": (
                    self._agent_session.session_id
                    if self._agent_session else None
                ),
            }
            pkl.dump(save_dict, f)
            logging.info(f"Saved approach to {save_path}_"
                         f"{online_learning_cycle}.AgentSDK")

    def load(self, online_learning_cycle: Optional[int] = None) -> None:
        """Load previously saved approach state."""
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.AgentSDK", "rb") as f:
            save_dict = pkl.load(f)

        self._processes = save_dict["processes"]
        self._learned_predicates = save_dict["learned_predicates"]
        self._offline_dataset = save_dict["offline_dataset"]
        self._online_dataset = save_dict["online_dataset"]
        self._online_learning_cycle = save_dict["online_learning_cycle"] + 1
        self._helper_types = save_dict.get("helper_types", set())
        self._augment_task_code = save_dict.get("augment_task_code", "")
        self._agent_proposed_options = save_dict.get(
            "agent_proposed_options", set())
        self._agent_proposed_processes = save_dict.get(
            "agent_proposed_processes", set())
        self._iteration_history = save_dict.get("iteration_history", [])
        self._agent_session_id = save_dict.get("agent_session_id")

        # Re-exec augment_task_code to restore the function
        if self._augment_task_code:
            exec_ctx = build_exec_context(
                self._types, self._get_current_predicates(),
                self._initial_options)
            result, error = exec_code_safely(
                self._augment_task_code, exec_ctx, "augment_task")
            if error:
                logging.warning(
                    f"Failed to restore augment_task function: {error}")
                self._augment_task_fn = None
            else:
                self._augment_task_fn = result

        # Restore types
        self._types = self._types | self._helper_types

        # Reseed options
        for proc in self._processes:
            if isinstance(proc, EndogenousProcess):
                proc.option.params_space.seed(CFG.seed)

        logging.info(
            f"Loaded {len(self._processes)} processes, "
            f"{len(self._learned_predicates)} learned predicates, "
            f"{len(self._offline_dataset.trajectories)} offline trajectories, "
            f"{len(self._online_dataset.trajectories)} online trajectories")
