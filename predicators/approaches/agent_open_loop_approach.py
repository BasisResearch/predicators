"""Model-free open-loop agent planning approach.

Combines online trajectory collection (via AgentExplorer) with open-loop
option plan generation (via Claude Agent SDK). No predicate/process/type
invention — just stores trajectories and generates plans.

Example command:
    python predicators/main.py --env pybullet_domino \
        --approach agent_open_loop --seed 0 \
        --num_train_tasks 1 --num_test_tasks 1 \
        --num_online_learning_cycles 1 --explorer agent
"""
import datetime
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.tools import create_inspection_only_mcp_tools
from predicators.approaches import ApproachFailure
from predicators.approaches.agent_session_mixin import AgentSessionMixin
from predicators.approaches.base_approach import BaseApproach
from predicators.explorers import create_explorer
from predicators.explorers.base_explorer import BaseExplorer
from predicators.settings import CFG
from predicators.structs import Action, Dataset, InteractionRequest, \
    InteractionResult, LowLevelTrajectory, ParameterizedOption, Predicate, \
    State, Task, Type


class AgentOpenLoopApproach(AgentSessionMixin, BaseApproach):
    """Model-free open-loop planning via Claude Agent SDK.

    - Collects trajectories online using AgentExplorer
    - At solve time, queries the agent for an option plan
    - No predicate/process/type invention
    """

    _log_subdir = "agent_open_loop"

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._offline_dataset = Dataset([])
        self._online_trajectories: List[LowLevelTrajectory] = []
        self._online_learning_cycle = 0
        self._requests_train_task_idxs: Optional[List[int]] = None

        # Create unique run identifier for this execution
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self._init_agent_session_state(
            types, initial_predicates, initial_options, train_tasks)

    @classmethod
    def get_name(cls) -> str:
        return "agent_open_loop"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_log_dir(self) -> str:
        """Override to create separate directory for each run."""
        base_log_dir = super()._get_log_dir()
        run_log_dir = os.path.join(base_log_dir, f"run_{self._run_id}")
        os.makedirs(run_log_dir, exist_ok=True)
        logging.info(f"Logging agent queries/responses to: {run_log_dir}")
        return run_log_dir

    # ------------------------------------------------------------------ #
    # AgentSessionMixin hooks
    # ------------------------------------------------------------------ #

    def _get_agent_model_name(self) -> str:
        return CFG.agent_open_loop_model_name

    def _get_agent_system_prompt(self) -> str:
        return (
            "You are a planning agent. You observe task environments through "
            "inspection tools and generate option plans to achieve goals. "
            "You have access to read-only tools to inspect types, predicates, "
            "options, trajectories, and training tasks. Use these to "
            "understand the environment and generate effective plans."
        )

    def _create_agent_mcp_tools(self) -> list:
        return create_inspection_only_mcp_tools(self._tool_context)

    def _get_agent_allowed_tools(self) -> Optional[List[str]]:
        tool_prefix = "mcp__predicator_tools__"
        return [
            f"{tool_prefix}inspect_options",
            f"{tool_prefix}inspect_trajectories",
            f"{tool_prefix}inspect_train_tasks",
        ]

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._offline_dataset = dataset
        self._tool_context.offline_trajectories = dataset.trajectories
        if dataset.trajectories:
            self._tool_context.example_state = \
                dataset.trajectories[0].states[0]

    def get_interaction_requests(self) -> List[InteractionRequest]:
        explorer = self._create_explorer()
        requests: List[InteractionRequest] = []
        self._requests_train_task_idxs = []
        for _ in range(CFG.online_nsrt_learning_requests_per_cycle):
            task_idx = self._rng.choice(len(self._train_tasks))
            policy, termination_function = explorer.get_exploration_strategy(
                task_idx, CFG.timeout)
            req = InteractionRequest(
                train_task_idx=task_idx,
                act_policy=policy,
                query_policy=lambda s: None,
                termination_function=termination_function)
            requests.append(req)
            self._requests_train_task_idxs.append(task_idx)
        return requests

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        assert self._requests_train_task_idxs is not None
        for i, result in enumerate(results):
            task_idx = self._requests_train_task_idxs[i]
            traj = LowLevelTrajectory(result.states,
                                      result.actions,
                                      _train_task_idx=task_idx)
            self._online_trajectories.append(traj)

        # Update tool context
        self._sync_tool_context()

        logging.info(
            f"[Run {self._run_id}] Cycle {self._online_learning_cycle}: "
            f"collected {len(results)} trajectories, "
            f"{len(self._online_trajectories)} total online.")

        self.save(self._online_learning_cycle)
        self._online_learning_cycle += 1

    # ------------------------------------------------------------------ #
    # Solving
    # ------------------------------------------------------------------ #

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        try:
            option_plan = self._query_agent_for_option_plan(task)
        except Exception as e:
            raise ApproachFailure(
                f"Agent failed to produce option plan: {e}")

        policy = utils.option_plan_to_policy(option_plan)

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _query_agent_for_option_plan(self, task: Task) -> list:
        """Query the agent for an option plan and parse it."""
        prompt = self._build_solve_prompt(task)
        responses = self._query_agent_sync(prompt)
        plan_text = self._extract_option_plan_text(responses)

        if not plan_text:
            raise ApproachFailure("Agent returned empty plan text.")

        return self._parse_and_ground_plan(plan_text, task)

    def _build_solve_prompt(self, task: Task) -> str:
        """Build the prompt for generating an option plan."""
        init_state = task.init
        objects = list(init_state)

        # Objects
        obj_strs = []
        for obj in sorted(objects, key=lambda o: o.name):
            obj_strs.append(f"  {obj.name}: {obj.type.name}")

        # Goal
        goal_strs = [str(a) for a in sorted(task.goal, key=str)]

        # Options
        option_strs = []
        for opt in sorted(self._initial_options, key=lambda o: o.name):
            type_sig = ", ".join(t.name for t in opt.types)
            params_dim = opt.params_space.shape[0]
            if params_dim > 0:
                low = opt.params_space.low.tolist()
                high = opt.params_space.high.tolist()
                param_info = (f", params_dim={params_dim}, "
                              f"low={low}, high={high}")
            else:
                param_info = ""
            option_strs.append(f"  {opt.name}({type_sig}{param_info})")

        # Current atoms
        atoms = utils.abstract(init_state, self._initial_predicates)
        atom_strs = [str(a) for a in sorted(atoms, key=str)]

        # Trajectory summary
        traj_summary = self._build_trajectory_summary()

        # State features (compact)
        state_str = init_state.dict_str(indent=2)

        prompt = f"""You are solving a task. Generate an option plan to achieve the goal.

## Goal
{chr(10).join(goal_strs)}

## Initial State Atoms
{chr(10).join(atom_strs)}

## Initial State Features
{state_str}

## Objects
{chr(10).join(obj_strs)}

## Available Options
{chr(10).join(option_strs)}
{traj_summary}
## Instructions
You can use the inspect tools to examine types, predicates, options, and past trajectories in more detail.

Based on the task information and any past trajectory data, output an option plan to achieve the goal.
Output the plan with one option per line in this exact format:
 OptionName(obj1:type1, obj2:type2)[param1, param2]

If an option has no continuous parameters, use empty brackets: OptionName(obj1:type1)[]

Output ONLY the option plan lines at the end, after any analysis."""

        return prompt

    def _build_trajectory_summary(self) -> str:
        """Summarize trajectory data for context."""
        all_trajs = (self._offline_dataset.trajectories +
                     self._online_trajectories)
        if not all_trajs:
            return ""

        max_trajs = CFG.agent_sdk_max_trajectories_in_context
        recent = all_trajs[-max_trajs:]
        lines = [f"\n## Trajectory Summary ({len(all_trajs)} total, "
                 f"showing last {len(recent)})"]

        for i, traj in enumerate(recent):
            n_steps = len(traj.actions)
            init_atoms = utils.abstract(traj.states[0],
                                        self._initial_predicates)
            final_atoms = utils.abstract(traj.states[-1],
                                         self._initial_predicates)
            new_atoms = final_atoms - init_atoms
            lost_atoms = init_atoms - final_atoms
            lines.append(f"\nTrajectory {i}: {n_steps} steps")
            if new_atoms:
                lines.append(f"  Gained: "
                             f"{', '.join(str(a) for a in sorted(new_atoms, key=str))}")
            if lost_atoms:
                lines.append(f"  Lost: "
                             f"{', '.join(str(a) for a in sorted(lost_atoms, key=str))}")

        return "\n".join(lines)

    def _extract_option_plan_text(
            self, responses: List[Dict[str, Any]]) -> str:
        """Extract plan text from agent responses."""
        text_parts = []
        for resp in responses:
            if resp.get("type") == "text":
                text_parts.append(resp.get("text", ""))
            elif "content" in resp:
                for block in resp.get("content", []):
                    if isinstance(block, dict) and \
                            block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
        return "\n".join(text_parts)

    def _parse_and_ground_plan(self, plan_text: str, task: Task) -> list:
        """Parse option plan text and ground into executable options."""
        objects = list(task.init)
        parsed = utils.parse_model_output_into_option_plan(
            plan_text, objects, self._types, self._initial_options,
            parse_continuous_params=True)
        if not parsed:
            raise ApproachFailure("Parsed empty option plan from agent.")

        grounded = []
        for option, objs, params in parsed:
            try:
                params_arr = ([] if len(params) == 0
                              else np.array(params, dtype=np.float32))
                ground_opt = option.ground(objs, params_arr)
                grounded.append(ground_opt)
            except Exception as e:
                logging.warning(f"[Run {self._run_id}] Failed to ground option "
                                f"{option.name}: {e}")
                break

        if not grounded:
            raise ApproachFailure("No options successfully grounded.")
        logging.info(f"[Run {self._run_id}] Agent produced plan with "
                     f"{len(grounded)} options.")
        return grounded

    # ------------------------------------------------------------------ #
    # Explorer
    # ------------------------------------------------------------------ #

    def _create_explorer(self) -> BaseExplorer:
        """Create explorer for interaction requests."""
        if CFG.explorer == "agent":
            self._sync_tool_context()
            return self._create_agent_explorer(
                self._initial_predicates, self._initial_options)
        return create_explorer(
            CFG.explorer,
            self._initial_predicates,
            self._initial_options,
            self._types,
            self._action_space,
            self._train_tasks,
        )

    def _sync_tool_context(self) -> None:
        """Synchronize ToolContext with current state."""
        self._tool_context.types = self._types
        self._tool_context.predicates = self._initial_predicates
        self._tool_context.options = self._initial_options
        self._tool_context.train_tasks = self._train_tasks
        self._tool_context.offline_trajectories = \
            self._offline_dataset.trajectories
        self._tool_context.online_trajectories = self._online_trajectories

        all_trajs = (self._offline_dataset.trajectories +
                     self._online_trajectories)
        if all_trajs:
            self._tool_context.example_state = all_trajs[0].states[0]

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def save(self, online_learning_cycle: Optional[int] = None) -> None:
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.AgentOpenLoop",
                  "wb") as f:
            save_dict = {
                "offline_dataset": self._offline_dataset,
                "online_trajectories": self._online_trajectories,
                "online_learning_cycle": self._online_learning_cycle,
                "run_id": self._run_id,
                "agent_session_id": (
                    self._agent_session.session_id
                    if self._agent_session else None
                ),
            }
            pkl.dump(save_dict, f)
            logging.info(f"[Run {self._run_id}] Saved approach to {save_path}_"
                         f"{online_learning_cycle}.AgentOpenLoop")

    def load(self, online_learning_cycle: Optional[int] = None) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.AgentOpenLoop",
                  "rb") as f:
            save_dict = pkl.load(f)

        self._offline_dataset = save_dict["offline_dataset"]
        self._online_trajectories = save_dict["online_trajectories"]
        self._online_learning_cycle = \
            save_dict["online_learning_cycle"] + 1
        self._agent_session_id = save_dict.get("agent_session_id")

        # Create new run_id for continued execution (each run gets own dir)
        # but log the original run_id for reference
        original_run_id = save_dict.get("run_id", "unknown")
        self._run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Re-sync tool context
        self._sync_tool_context()

        logging.info(
            f"[Run {self._run_id}] Loaded from previous run {original_run_id}: "
            f"{len(self._offline_dataset.trajectories)} offline, "
            f"{len(self._online_trajectories)} online trajectories")
