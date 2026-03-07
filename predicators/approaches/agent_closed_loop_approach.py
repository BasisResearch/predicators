"""Agent closed-loop planning approach.

Like AgentPlannerApproach, but instead of generating the full option plan
upfront, the agent is queried at each option boundary to decide the next
single option based on the current state. This makes the approach reactive
to actual execution outcomes.

Example command:
    python predicators/main.py --env pybullet_domino \
        --approach agent_closed_loop --seed 0 \
        --num_train_tasks 1 --num_test_tasks 1 \
        --num_online_learning_cycles 1 --explorer agent
"""
import logging
from typing import Callable, List

import numpy as np

from predicators import utils
from predicators.agent_sdk.tools import create_mcp_tools
from predicators.approaches import ApproachFailure
from predicators.approaches.agent_planner_approach import AgentPlannerApproach
from predicators.settings import CFG
from predicators.structs import Action, State, Task, _Option


class AgentClosedLoopApproach(AgentPlannerApproach):
    """Closed-loop planning via Claude Agent SDK.

    At each option boundary, queries the agent for the next single
    option based on the current state, goal, and execution history.
    """

    @classmethod
    def get_name(cls) -> str:
        return "agent_closed_loop"

    def _create_agent_mcp_tools(self) -> list:
        return create_mcp_tools(
            self._tool_context,
            tool_names=[
                "inspect_options", "inspect_trajectories",
                "inspect_train_tasks"
            ],
        )

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        step_history: List[str] = []

        def _option_policy(state: State) -> _Option:
            try:
                prompt = self._build_step_prompt(state, task, step_history)
                responses = self._query_agent_sync(prompt)
                text = self._extract_option_plan_text(responses)
                option = self._parse_single_option(text, task)
                step_history.append(option.simple_str())
                return option
            except ApproachFailure:
                raise
            except Exception as e:
                raise ApproachFailure(
                    f"Agent failed to produce next option: {e}")

        policy = utils.option_policy_to_policy(_option_policy)

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _build_step_prompt(self, state: State, task: Task,
                           step_history: List[str]) -> str:
        """Build prompt asking for the next single option."""
        objects = list(state)

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
                if opt.params_description:
                    desc = ", ".join(opt.params_description)
                    param_info = (f", params=[{desc}], "
                                  f"low={low}, high={high}")
                else:
                    param_info = (f", params_dim={params_dim}, "
                                  f"low={low}, high={high}")
            else:
                param_info = ""
            option_strs.append(f"  {opt.name}({type_sig}{param_info})")

        # Current atoms
        atoms = utils.abstract(state, self._initial_predicates)
        atom_strs = [str(a) for a in sorted(atoms, key=str)]

        # State features
        state_str = state.dict_str(indent=2)

        # Trajectory summary
        traj_summary = self._build_trajectory_summary()

        # Step history
        if step_history:
            history_str = "\n## Options Executed So Far\n"
            for i, s in enumerate(step_history):
                history_str += f"  Step {i + 1}: {s}\n"
        else:
            history_str = "\n## Options Executed So Far\nNone yet (this is the first step).\n"

        prompt = f"""You are solving a task step by step. Decide the NEXT SINGLE option to execute.

## Goal
{chr(10).join(goal_strs)}

## Current State Atoms
{chr(10).join(atom_strs)}

## Current State Features
{state_str}

## Objects
{chr(10).join(obj_strs)}

## Available Options
{chr(10).join(option_strs)}
{history_str}{traj_summary}
## Instructions
You can use the inspect tools to examine types, predicates, options, and past trajectories in more detail.

Based on the current state and execution history, output the NEXT SINGLE option to execute.
Output exactly ONE option line in this format:
 OptionName(obj1:type1, obj2:type2)[param1, param2]

If an option has no continuous parameters, use empty brackets: OptionName(obj1:type1)[]

Output ONLY the single option line at the end, after any analysis."""

        return prompt

    def _parse_single_option(self, text: str, task: Task) -> _Option:
        """Parse a single option from agent response and ground it."""
        if not text.strip():
            raise ApproachFailure("Agent returned empty response.")

        objects = list(task.init)
        parsed = utils.parse_model_output_into_option_plan(
            text,
            objects,
            self._types,
            self._initial_options,
            parse_continuous_params=True)

        if not parsed:
            raise ApproachFailure(
                "Could not parse any option from agent response.")

        # Take the last parsed option (agent may include analysis before it)
        option, objs, params = parsed[-1]
        try:
            params_arr = np.array(params, dtype=np.float32)
            ground_opt = option.ground(objs, params_arr)
        except Exception as e:
            raise ApproachFailure(
                f"Failed to ground option {option.name}: {e}")

        logging.info(f"Agent selected next option: "
                     f"{ground_opt.simple_str()}")
        return ground_opt
