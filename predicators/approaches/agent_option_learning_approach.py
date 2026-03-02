"""Agent option learning approach: online option invention via Claude Agent
SDK.

Uses a persistent Claude agent session to iteratively propose/retract
parameterized options based on observed trajectory data, then uses those
options for open-loop planning.  No predicate/process/type invention.

Example command::

    python predicators/main.py --env pybullet_domino \\
        --approach agent_option_learning --seed 0 \\
        --num_train_tasks 1 --num_test_tasks 1 \\
        --num_online_learning_cycles 3 --explorer agent
"""
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import dill as pkl
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.option_builder import build_option_helpers
from predicators.agent_sdk.proposal_parser import ProposalBundle, \
    build_exec_context, exec_code_safely
from predicators.approaches.agent_planner_approach import AgentPlannerApproach
from predicators.explorers.base_explorer import BaseExplorer
from predicators.settings import CFG
from predicators.structs import Action, InteractionResult, \
    LowLevelTrajectory, ParameterizedOption, Predicate, State, Task, Type


class AgentOptionLearningApproach(AgentPlannerApproach):
    """Option-learning planning approach using Claude Agent SDK.

    Extends AgentPlannerApproach with the ability to invent and retract
    parameterized options during online learning.  The agent writes
    Python code that uses high-level option-builder helpers to define
    new options.
    """

    _log_subdir = "agent_option_learning"

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        # Agent-specific state (before super().__init__)
        self._agent_proposed_options: Set[ParameterizedOption] = set()
        self._option_proposal_code: Dict[str, str] = {}  # name -> code
        self._iteration_history: List[Dict[str, Any]] = []

        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)

    @classmethod
    def get_name(cls) -> str:
        return "agent_option_learning"

    # ------------------------------------------------------------------ #
    # AgentSessionMixin hooks
    # ------------------------------------------------------------------ #

    def _get_agent_model_name(self) -> str:
        return CFG.agent_option_learning_model_name

    def _get_agent_system_prompt(self) -> str:
        return (
            "You are an option inventor for a robot planning system. Your "
            "role is to propose new parameterized options (high-level "
            "actions) that help achieve task goals, based on observed "
            "trajectory data.\n\n"
            "## What You Observe\n"
            "- Trajectory data: sequences of states and actions\n"
            "- Task goals: symbolic goal descriptions\n"
            "- Current options: parameterized options currently available\n"
            "- Iteration history: outcomes of previous proposals\n\n"
            "## What You Can Do\n"
            "1. **Propose options**: Write Python code defining "
            "`proposed_options` (a list of ParameterizedOption objects)\n"
            "2. **Retract options**: Remove previously proposed options "
            "that aren't working\n"
            "3. **Test option plans**: Execute option sequences to verify "
            "they work\n\n"
            "## Option Building Helpers (available in exec context)\n"
            "- `make_move_to_pose_option(name, types, params_space, "
            "pose_fn, finger_status)` — move EE to a target pose.  "
            "`pose_fn(state, objects, params) -> (x, y, z, roll, pitch, "
            "yaw)`\n"
            "- `make_finger_option(name, types, params_space, mode)` — "
            "open/close fingers.  `mode`: 'open' or 'close'\n"
            "- `chain_options(name, children)` — chain options into a "
            "sequence\n"
            "All standard imports (np, Box, ParameterizedOption, State, "
            "Type, etc.) and current types/options are available by name "
            "in the exec context.\n\n"
            "## Iteration Protocol\n"
            "1. Inspect trajectory data and current options\n"
            "2. Hypothesise what new options would help achieve task goals\n"
            "3. Propose options via the propose_options tool\n"
            "4. Test your proposals using test_option_plan\n"
            "5. Retract options that don't work, propose improved versions")

    def _get_agent_tool_names(self) -> Optional[List[str]]:
        return [
            "inspect_types",
            "inspect_options",
            "inspect_trajectories",
            "inspect_train_tasks",
            "inspect_iteration_history",
            "propose_options",
            "retract_abstractions",
            "test_option_plan",
        ]

    # ------------------------------------------------------------------ #
    # Overridable helpers (from AgentPlannerApproach)
    # ------------------------------------------------------------------ #

    def _get_all_options(self) -> Set[ParameterizedOption]:
        return self._initial_options | self._agent_proposed_options

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Disable proposals during test-time solving.

        Session isolation is handled by the parent AgentPlannerApproach._solve.
        """
        self._tool_context.proposals_disabled = True
        try:
            return super()._solve(task, timeout)
        finally:
            self._tool_context.proposals_disabled = False

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        """Learn from interaction results: collect trajectories, then run an
        agent iteration to propose/retract options."""
        # 1. Convert results to trajectories
        assert self._requests_train_task_idxs is not None
        for i, result in enumerate(results):
            task_idx = self._requests_train_task_idxs[i]
            traj = LowLevelTrajectory(result.states,
                                      result.actions,
                                      _train_task_idx=task_idx)
            self._online_trajectories.append(traj)

        # 2. Sync tool context (including proposed options + builder helpers)
        self._sync_tool_context()

        # 3. Run agent iteration
        self._run_agent_iteration()

        # 4. Integrate proposals (options only)
        proposals = self._tool_context.iteration_proposals
        self._integrate_proposals(proposals)

        # 5. Log iteration summary
        summary = self._build_iteration_summary(proposals)
        self._iteration_history.append(summary)
        self._tool_context.iteration_history = self._iteration_history
        logging.info(f"[Run {self._run_id}] Iteration "
                     f"{self._online_learning_cycle} summary: "
                     f"{json.dumps(summary, default=str)}")

        # 6. Save iteration logs, save state, increment cycle
        self._save_iteration_logs(self._online_learning_cycle)
        self.save(self._online_learning_cycle)
        self._online_learning_cycle += 1

    def _sync_tool_context(self) -> None:
        """Synchronize ToolContext with current state."""
        # Call parent sync
        super()._sync_tool_context()

        # Override options to include agent-proposed ones
        self._tool_context.options = (self._initial_options
                                      | self._agent_proposed_options)

        # Inject option builder helpers
        self._tool_context.option_builder_context = build_option_helpers(
            CFG.env)

        # Reset proposals for this iteration
        self._tool_context.iteration_proposals = ProposalBundle()

    def _run_agent_iteration(self) -> None:
        """Build iteration message and query the Claude agent."""
        self._ensure_agent_session()

        all_trajs = self._get_all_trajectories()
        num_total = len(all_trajs)
        task_success = self._compute_task_success_rate(all_trajs)

        # Current options summary
        all_opts = self._get_all_options()
        opt_strs = []
        for opt in sorted(all_opts, key=lambda o: o.name):
            type_sig = ", ".join(t.name for t in opt.types)
            params_dim = opt.params_space.shape[0]
            src = "initial" if opt in self._initial_options else "proposed"
            opt_strs.append(
                f"  {opt.name}({type_sig}), params_dim={params_dim} "
                f"[{src}]")

        # Previous iteration outcomes
        prev_outcomes = (
            "No previous iterations." if not self._iteration_history else
            json.dumps(self._iteration_history[-1], default=str, indent=2))

        # Trajectory summary
        traj_summary = self._build_trajectory_summary()

        message = f"""## Iteration {self._online_learning_cycle}

### Data
- Total trajectories: {num_total}
- Task success rate: {task_success:.1%}
{traj_summary}

### Current Options ({len(all_opts)})
{chr(10).join(opt_strs)}

### Previous Iteration Outcomes
{prev_outcomes}

### Available Tools
  - inspect_types, inspect_options, inspect_trajectories, inspect_train_tasks
  - inspect_iteration_history
  - propose_options (code defining `proposed_options`)
  - retract_abstractions (remove options by name)
  - test_option_plan (execute option sequences)

### Instructions
Analyse the trajectory data and task goals.  Propose new options that would \
help achieve the goals, or retract options that are not working.  Use the \
option builder helpers (make_move_to_pose_option, make_finger_option, \
chain_options) in your code.  Test your proposed plans \
before committing."""

        self._last_context_message = message
        self._last_agent_responses = self._query_agent_sync(message)

    def _integrate_proposals(self, proposals: ProposalBundle) -> None:
        """Integrate validated option proposals into approach state."""
        # Add new options
        if proposals.proposed_options:
            self._agent_proposed_options |= proposals.proposed_options
            names = [o.name for o in proposals.proposed_options]
            logging.info(f"[Run {self._run_id}] Integrated "
                         f"{len(proposals.proposed_options)} new options: "
                         f"{names}")

        # Retract options by name
        if proposals.retract_option_names:
            before = len(self._agent_proposed_options)
            removed_names = set()
            remaining = set()
            for opt in self._agent_proposed_options:
                if opt.name in proposals.retract_option_names:
                    removed_names.add(opt.name)
                    # Also remove saved code
                    self._option_proposal_code.pop(opt.name, None)
                else:
                    remaining.add(opt)
            self._agent_proposed_options = remaining
            logging.info(
                f"[Run {self._run_id}] Retracted "
                f"{before - len(self._agent_proposed_options)} options: "
                f"{sorted(removed_names)}")

    def _compute_task_success_rate(self,
                                   trajs: List[LowLevelTrajectory]) -> float:
        """Compute fraction of trajectories that achieved their task goal."""
        if not trajs:
            return 0.0
        successes = 0
        counted = 0
        for traj in trajs:
            if (traj._train_task_idx is not None
                    and traj._train_task_idx < len(self._train_tasks)):
                task = self._train_tasks[traj._train_task_idx]
                goal_preds = {a.predicate for a in task.goal}
                final_atoms = utils.abstract(traj.states[-1], goal_preds)
                if task.goal.issubset(final_atoms):
                    successes += 1
                counted += 1
        return successes / max(counted, 1)

    def _build_iteration_summary(self,
                                 proposals: ProposalBundle) -> Dict[str, Any]:
        """Build a summary dict of what happened this iteration."""
        return {
            "cycle": self._online_learning_cycle,
            "proposed_options": [o.name for o in proposals.proposed_options],
            "retracted_options": sorted(proposals.retract_option_names),
            "errors": proposals.errors,
            "total_options": len(self._get_all_options()),
            "total_agent_proposed": len(self._agent_proposed_options),
        }

    # ------------------------------------------------------------------ #
    # Explorer
    # ------------------------------------------------------------------ #

    def _create_explorer(self) -> BaseExplorer:
        """Create explorer, passing agent-proposed options."""
        if CFG.explorer == "agent":
            self._sync_tool_context()
            return self._create_agent_explorer(
                self._initial_predicates,
                self._initial_options | self._agent_proposed_options)
        return super()._create_explorer()

    # ------------------------------------------------------------------ #
    # Iteration logs
    # ------------------------------------------------------------------ #

    def _save_iteration_logs(self, cycle: int) -> None:
        """Save iteration-specific logs to disk."""
        log_dir = os.path.join(self._get_log_dir(), f"iteration_{cycle}")
        os.makedirs(log_dir, exist_ok=True)

        # Context message
        if hasattr(self, '_last_context_message'):
            with open(os.path.join(log_dir, "context_message.txt"), "w") as f:
                f.write(self._last_context_message)

        # Agent responses
        if (CFG.agent_sdk_log_agent_responses
                and hasattr(self, '_last_agent_responses')):
            resp_path = os.path.join(log_dir, "agent_responses.jsonl")
            with open(resp_path, "w") as f:
                for resp in self._last_agent_responses:
                    f.write(json.dumps(resp, default=str) + "\n")

        # Proposed options
        proposals = self._tool_context.iteration_proposals
        if proposals.proposed_options:
            opts_dir = os.path.join(log_dir, "proposed_options")
            os.makedirs(opts_dir, exist_ok=True)
            with open(os.path.join(opts_dir, "options.json"), "w") as f:
                json.dump([o.name for o in proposals.proposed_options],
                          f,
                          indent=2)

        # Retractions
        if proposals.retract_option_names:
            with open(os.path.join(log_dir, "retractions.json"), "w") as f:
                json.dump({"options": sorted(proposals.retract_option_names)},
                          f,
                          indent=2)

        # Current full option inventory
        all_opts = self._get_all_options()
        with open(os.path.join(log_dir, "all_options.json"), "w") as f:
            json.dump(
                {
                    "initial":
                    sorted(o.name for o in self._initial_options),
                    "agent_proposed":
                    sorted(o.name for o in self._agent_proposed_options),
                },
                f,
                indent=2)

        # Session info
        if self._agent_session is not None:
            self._agent_session.save_session_info()

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def save(self, online_learning_cycle: Optional[int] = None) -> None:
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.AgentOptionLearning",
                  "wb") as f:
            save_dict = {
                "offline_dataset":
                self._offline_dataset,
                "online_trajectories":
                self._online_trajectories,
                "online_learning_cycle":
                self._online_learning_cycle,
                "run_id":
                self._run_id,
                "agent_proposed_options":
                self._agent_proposed_options,
                "option_proposal_code":
                self._option_proposal_code,
                "iteration_history":
                self._iteration_history,
                "agent_session_id": (self._agent_session.session_id
                                     if self._agent_session else None),
            }
            pkl.dump(save_dict, f)
            logging.info(f"[Run {self._run_id}] Saved approach to {save_path}_"
                         f"{online_learning_cycle}.AgentOptionLearning")

    def load(self, online_learning_cycle: Optional[int] = None) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.AgentOptionLearning",
                  "rb") as f:
            save_dict = pkl.load(f)

        self._offline_dataset = save_dict["offline_dataset"]
        self._online_trajectories = save_dict["online_trajectories"]
        self._online_learning_cycle = \
            save_dict["online_learning_cycle"] + 1
        self._agent_session_id = save_dict.get("agent_session_id")
        self._iteration_history = save_dict.get("iteration_history", [])
        self._option_proposal_code = save_dict.get("option_proposal_code", {})

        # Restore run_id
        import datetime
        original_run_id = save_dict.get("run_id", "unknown")
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Try to restore agent-proposed options from saved objects
        # (dill should handle the closures), falling back to re-exec
        self._agent_proposed_options = save_dict.get("agent_proposed_options",
                                                     set())

        # If dill restoration failed, try re-executing code
        if not self._agent_proposed_options and self._option_proposal_code:
            helpers = build_option_helpers(CFG.env)
            exec_ctx = build_exec_context(self._types,
                                          self._initial_predicates,
                                          self._initial_options,
                                          extra_context=helpers)
            for opt_name, code in self._option_proposal_code.items():
                result, error = exec_code_safely(code, exec_ctx,
                                                 "proposed_options")
                if error:
                    logging.warning(
                        f"Failed to restore option '{opt_name}': {error}")
                elif isinstance(result, (list, set)):
                    self._agent_proposed_options |= set(result)

        # Re-sync tool context
        self._sync_tool_context()

        logging.info(
            f"[Run {self._run_id}] Loaded from previous run "
            f"{original_run_id}: "
            f"{len(self._offline_dataset.trajectories)} offline, "
            f"{len(self._online_trajectories)} online trajectories, "
            f"{len(self._agent_proposed_options)} agent-proposed options")
