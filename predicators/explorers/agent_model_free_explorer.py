"""Agent model-free explorer: Claude agent generates grounded option plans
without a learned world model.

Produces fully-grounded option plans (including continuous parameters)
from a one-shot task description and rolls them out in the real
environment. Unlike ``AgentModelBasedExplorer``, the agent has no belief
simulator to validate against and no backtracking refinement; it must
supply complete parameters itself.

Registered under the CLI explorer name ``agent_model_free``
(``agent_plan`` is kept as a deprecated alias).
"""

import logging

import numpy as np

from predicators import utils
from predicators.agent_sdk.session_base import AgentSessionFatalError, \
    query_fatal_error
from predicators.agent_sdk.session_manager import run_query_sync
from predicators.explorers.agent_explorer_base import AgentExplorerBase
from predicators.settings import CFG
from predicators.structs import ExplorationStrategy, Task


class AgentModelFreeExplorer(AgentExplorerBase):
    """Queries a Claude agent to produce grounded option plans."""

    @classmethod
    def get_name(cls) -> str:
        return "agent_model_free"

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        task = self._train_tasks[train_task_idx]
        try:
            prompt = self._build_exploration_prompt(train_task_idx)
            responses = run_query_sync(self._agent_session,
                                       prompt,
                                       kind="explore")
            dead = query_fatal_error(responses)
            if dead is not None:
                # See agent_model_based_explorer: never explore at random
                # because the session backend is down.
                raise AgentSessionFatalError(
                    "explore query died without the agent doing any work "
                    f"({dead}); not falling back to random exploration.")
            plan_text = self._extract_option_plan_text(responses)
            if plan_text:
                option_plan = self._parse_and_ground_plan(plan_text, task)
                if option_plan:
                    # The Wait wrapper needs the abstraction to end a
                    # Wait on its target atoms.
                    policy = utils.option_plan_to_policy(
                        option_plan,
                        abstract_function=lambda s: utils.abstract(
                            s, self._predicates))
                    return policy, lambda _: False
            logging.info("Agent explorer: no valid plan, falling back to "
                         "random options.")
        except AgentSessionFatalError:
            # A random fallback would hide the broken session backend;
            # re-raise so the run terminates.
            raise
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(f"Agent explorer failed: {e}. "
                            "Falling back to random options.")

        if not CFG.agent_explorer_fallback_to_random:
            raise utils.RequestActPolicyFailure(
                "Agent explorer failed and fallback disabled.")
        return self._random_options_fallback()

    def _build_exploration_prompt(self, train_task_idx: int) -> str:
        """Build a prompt for the agent to produce an option plan."""
        task = self._train_tasks[train_task_idx]
        init_state = task.init

        objects = list(init_state)
        obj_strs = []
        for obj in sorted(objects, key=lambda o: o.name):
            obj_strs.append(f"  {obj.name}: {obj.type.name}")

        # Goal atoms
        goal_strs = [str(a) for a in sorted(task.goal, key=str)]

        # Available options with signatures.
        all_options = self._options
        option_strs = []
        for opt in sorted(all_options, key=lambda o: o.name):
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
        atoms = utils.abstract(init_state, self._predicates)
        atom_strs = [str(a) for a in sorted(atoms, key=str)]

        # Trajectory summary
        traj_summary = self._build_trajectory_summary()

        # Available tools
        tools_str = ""
        tool_names = self._agent_tool_names()
        if tool_names:
            tool_list = "\n".join(f"  - {t}" for t in tool_names)
            tools_str = f"\n## Available Tools\n{tool_list}\n"

        task_intro = ("You are exploring a task environment. "
                      f"Generate an option plan to explore task "
                      f"{train_task_idx}.")
        prompt = f"""{task_intro}

## Goal
{chr(10).join(goal_strs)}

## Initial State Atoms
{chr(10).join(atom_strs)}

## Objects
{chr(10).join(obj_strs)}

## Available Options
{chr(10).join(option_strs)}
{traj_summary}{self._world_model_notes_block()}{tools_str}
## Instructions
Use your available tools to inspect the environment and test your plan before committing to it.

Output an option plan, one option per line, in this exact format:
OptionName(obj1:type1, obj2:type2)[param1, param2]

If an option has no continuous parameters, use empty brackets: OptionName(obj1:type1)[]

Output ONLY the option plan lines at the end, after any analysis."""

        return prompt

    def _parse_and_ground_plan(self, plan_text: str, task: Task) -> list:
        """Parse option plan text and ground into executable options."""
        objects = list(task.init)
        all_options = self._options
        parsed = utils.parse_model_output_into_option_plan(
            plan_text,
            objects,
            self._types,
            all_options,
            parse_continuous_params=True)
        if not parsed:
            logging.info("Agent explorer: parsed empty option plan.")
            return []

        grounded = []
        for option, objs, params in parsed:
            try:
                ground_opt = option.ground(objs,
                                           np.array(params, dtype=np.float32))
                grounded.append(ground_opt)
            except Exception as e:  # pylint: disable=broad-except
                logging.info(f"Agent explorer: failed to ground "
                             f"option {option.name}: {e}")
                break

        if not grounded:
            logging.info("Agent explorer: no options successfully grounded.")
        else:
            logging.info(f"Agent explorer: grounded {len(grounded)} options.")
        return grounded
