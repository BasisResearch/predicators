"""Agent abstraction learning approach: online process and predicate invention.

Uses a persistent Claude Agent SDK session to iteratively propose
abstractions (types, predicates, helper objects, processes, options)
based on observed trajectory data and planning feedback.
"""
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import dill as pkl
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.agent_session_mixin import AgentSessionMixin
from predicators.agent_sdk.proposal_parser import ProposalBundle, \
    build_exec_context, exec_code_safely
from predicators.approaches.agent_planner_approach import AgentPlannerApproach
from predicators.approaches.pp_online_process_learning_approach import \
    OnlineProcessLearningAndPlanningApproach
from predicators.approaches.pp_predicate_invention_approach import \
    PredicateInventionProcessPlanningApproach
from predicators.explorers.base_explorer import BaseExplorer
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.settings import CFG
from predicators.structs import Action, CausalProcess, Dataset, \
    EndogenousProcess, InteractionResult, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type


class AgentAbstractionLearningApproach(
        AgentPlannerApproach, PredicateInventionProcessPlanningApproach,
        OnlineProcessLearningAndPlanningApproach):
    """Abstraction-learning planning approach using Claude Agent SDK.

    Maintains a persistent Claude agent session that iteratively refines
    abstraction proposals based on observed trajectory data and planning
    feedback. The agent cannot see environment source code -- it
    observes the world only through custom MCP tools.
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
        self._last_context_message: str = ""
        self._last_agent_responses: List[Any] = []
        self._agent_session_id: Optional[str] = None
        self._option_model = create_option_model(CFG.option_model_name)

        self._init_agent_session_state(types, initial_predicates,
                                       initial_options, train_tasks)

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
        return "agent_abstraction_learning"

    # ------------------------------------------------------------------ #
    # AgentSessionMixin hooks
    # ------------------------------------------------------------------ #

    def _get_log_dir(self) -> str:
        """Use the mixin's simple log dir (no run_id subdirectory)."""
        # pylint: disable-next=protected-access
        return AgentSessionMixin._get_log_dir(self)

    def _get_agent_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    # ------------------------------------------------------------------ #
    # Overridable helpers (from AgentPlannerApproach)
    # ------------------------------------------------------------------ #

    def _get_all_options(self) -> Set[ParameterizedOption]:
        return self._initial_options | self._agent_proposed_options

    def _get_all_predicates(self) -> Set[Predicate]:
        return self._get_current_predicates()

    def _get_all_trajectories(self) -> list:
        return (self._offline_dataset.trajectories +
                self._online_dataset.trajectories)

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Store the offline dataset.

        Do NOT start agent session yet.
        """
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

    # pylint: disable-next=arguments-differ
    def _sync_tool_context(  # type: ignore[override]
            self, all_trajs: List[LowLevelTrajectory]) -> None:
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
        self._tool_context.option_model = self._option_model
        self._tool_context.iteration_id = self._online_learning_cycle

        if all_trajs:
            self._tool_context.example_state = all_trajs[0].states[0]

        # Reset proposals for this iteration
        self._tool_context.iteration_proposals = ProposalBundle()

    def _run_agent_iteration(self,
                             all_trajs: List[LowLevelTrajectory]) -> None:
        """Build iteration message and query the Claude agent."""
        self._ensure_agent_session()

        # Build the iteration message
        num_new = len(self._online_dataset.trajectories)
        num_total = len(all_trajs)
        task_success = self._compute_task_success_rate(all_trajs)

        type_str = ", ".join(
            f"{t.name}[{','.join(t.feature_names)}]"
            for t in sorted(self._types, key=lambda t: t.name))
        preds = self._get_current_predicates()
        pred_str = ", ".join(f"{p.name}({','.join(t.name for t in p.types)})"
                             for p in sorted(preds, key=lambda p: p.name))
        procs = self._get_current_processes()
        proc_str = ", ".join(p.name
                             for p in sorted(procs, key=lambda p: p.name))
        opt_str = ", ".join(
            o.name
            for o in sorted(self._initial_options, key=lambda o: o.name))

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
            available_tools=self._agent_session.tool_names
            if self._agent_session else None,
        )

        # Save the context message
        self._last_context_message = message

        # Run async query via mixin helper
        self._last_agent_responses = self._query_agent_sync(message)

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
            logging.info(f"Integrated {len(proposals.proposed_processes)} "
                         f"new processes (total: "
                         f"{len(self._get_current_processes())})")

        # Options
        if proposals.proposed_options:
            self._agent_proposed_options |= proposals.proposed_options
            logging.info(f"Integrated {len(proposals.proposed_options)} "
                         f"new options")

        # Retractions
        if proposals.retract_type_names:
            removed = {
                t
                for t in self._helper_types
                if t.name in proposals.retract_type_names
            }
            self._helper_types -= removed
            self._types -= removed
            logging.info(f"Retracted {len(removed)} helper types: "
                         f"{[t.name for t in removed]}")

        if proposals.retract_predicate_names:
            before = len(self._learned_predicates)
            self._learned_predicates = {
                p
                for p in self._learned_predicates
                if p.name not in proposals.retract_predicate_names
            }
            logging.info(
                f"Retracted "
                f"{before - len(self._learned_predicates)} predicates")

        if proposals.retract_object_augmentor:
            self._augment_task_fn = None
            self._augment_task_code = ""
            logging.info("Retracted object augmentor")

        if proposals.retract_process_names:
            before = len(self._agent_proposed_processes)
            self._agent_proposed_processes = {
                p
                for p in self._agent_proposed_processes
                if p.name not in proposals.retract_process_names
            }
            logging.info(f"Retracted "
                         f"{before - len(self._agent_proposed_processes)} "
                         f"processes")

        if proposals.retract_option_names:
            before = len(self._agent_proposed_options)
            self._agent_proposed_options = {
                o
                for o in self._agent_proposed_options
                if o.name not in proposals.retract_option_names
            }
            logging.info(f"Retracted "
                         f"{before - len(self._agent_proposed_options)} "
                         f"options")

    def _get_current_processes(self) -> Set[CausalProcess]:
        """Get current processes including agent-proposed ones."""
        return self._processes | self._agent_proposed_processes

    def _compute_task_success_rate(self,
                                   trajs: List[LowLevelTrajectory]) -> float:
        """Compute fraction of trajectories that achieved their task goal."""
        if not trajs:
            return 0.0
        successes = 0
        counted = 0
        for traj in trajs:
            idx = traj._train_task_idx  # pylint: disable=protected-access
            if idx is not None and \
                    idx < len(self._train_tasks):
                task = self._train_tasks[idx]
                goal_preds = {a.predicate for a in task.goal}
                final_atoms = utils.abstract(traj.states[-1], goal_preds)
                if task.goal.issubset(final_atoms):
                    successes += 1
                counted += 1
        return successes / max(counted, 1)

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Solve via agent-driven option plan generation."""
        if self._augment_task_fn is not None:
            try:
                task = self._augment_task_fn(task)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning(f"Task augmentation failed: {e}. "
                                f"Using original task.")

        all_trajs = self._get_all_trajectories()
        self._tool_context.current_task = task
        self._sync_tool_context(all_trajs)
        try:
            return super()._solve(task, timeout)
        finally:
            self._tool_context.current_task = None

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

        # Options (include agent-proposed)
        option_strs = []
        for opt in sorted(self._get_all_options(), key=lambda o: o.name):
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

        # Current atoms (include learned predicates)
        atoms = utils.abstract(init_state, self._get_all_predicates())
        atom_strs = [str(a) for a in sorted(atoms, key=str)]

        # Trajectory summary
        traj_summary = self._build_trajectory_summary()

        # State features
        state_str = init_state.dict_str(indent=2)

        # Processes summary
        procs = self._get_current_processes()
        proc_strs = []
        for proc in sorted(procs, key=lambda p: p.name):
            conds = ", ".join(str(a) for a in sorted(proc.condition_at_start))
            adds = ", ".join(str(a) for a in sorted(proc.add_effects))
            dels = ", ".join(str(a) for a in sorted(proc.delete_effects))
            proc_strs.append(f"  {proc.name}: conds={{{conds}}}, "
                             f"add={{{adds}}}, del={{{dels}}}")

        proc_section = ""
        if proc_strs:
            proc_section = (f"\n## Processes ({len(procs)})\n" +
                            "\n".join(proc_strs) + "\n")

        prompt = f"""You are solving a task. Generate an option plan \
to achieve the goal.

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
{proc_section}{traj_summary}
## Available Tools
You have access to planning tools:
  - generate_bilevel_plan: Get a complete plan with sampled params from the bilevel planner
  - generate_abstract_plan: Get a plan skeleton with parameter space info
  - test_option_plan: Test an option plan on the current task
  - inspect_trajectories, inspect_options, inspect_predicates, etc.

## Instructions
Use your available tools to generate and test plans before committing.

Recommended workflow:
1. Call generate_bilevel_plan (no task_idx needed - uses current task) to get a baseline plan
2. Optionally call test_option_plan to verify the plan works
3. Adjust parameters if needed and test again

Output the final plan with one option per line in this exact format:
 OptionName(obj1:type1, obj2:type2)[param1, param2]

If an option has no continuous parameters, use empty brackets: OptionName(obj1:type1)[]

Output ONLY the option plan lines at the end, after any analysis."""

        return prompt

    # ------------------------------------------------------------------ #
    # Explorer
    # ------------------------------------------------------------------ #

    def _create_explorer(self) -> BaseExplorer:
        """Create explorer, passing agent context if using agent explorer."""
        if CFG.explorer == "agent_plan":
            all_trajs = (self._offline_dataset.trajectories +
                         self._online_dataset.trajectories)
            self._sync_tool_context(all_trajs)
            preds = self._get_current_predicates()
            return self._create_agent_explorer(
                preds, self._initial_options | self._agent_proposed_options)
        return super()._create_explorer()

    # ------------------------------------------------------------------ #
    # Iteration summary / logs
    # ------------------------------------------------------------------ #

    def _build_iteration_summary(self,
                                 proposals: ProposalBundle) -> Dict[str, Any]:
        """Build a summary dict of what happened this iteration."""
        return {
            "cycle": self._online_learning_cycle,
            "proposed_types": [t.name for t in proposals.proposed_types],
            "proposed_predicates":
            [p.name for p in proposals.proposed_predicates],
            "proposed_augmentor": proposals.augment_task_code is not None,
            "proposed_processes":
            [p.name for p in proposals.proposed_processes],
            "proposed_options": [o.name for o in proposals.proposed_options],
            "retracted_types": sorted(proposals.retract_type_names),
            "retracted_predicates": sorted(proposals.retract_predicate_names),
            "retracted_augmentor": proposals.retract_object_augmentor,
            "retracted_processes": sorted(proposals.retract_process_names),
            "retracted_options": sorted(proposals.retract_option_names),
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
            with open(os.path.join(log_dir, "context_message.txt"),
                      "w",
                      encoding="utf-8") as f:
                f.write(self._last_context_message)

        # Agent responses
        if CFG.agent_sdk_log_agent_responses and \
                hasattr(self, '_last_agent_responses'):
            resp_path = os.path.join(log_dir, "agent_responses.jsonl")
            with open(resp_path, "w", encoding="utf-8") as f:
                for resp in self._last_agent_responses:
                    f.write(json.dumps(resp, default=str) + "\n")

        # Proposals directory
        proposals_dir = os.path.join(log_dir, "proposals")
        os.makedirs(proposals_dir, exist_ok=True)

        proposals = self._tool_context.iteration_proposals
        if proposals.proposed_types:
            with open(os.path.join(proposals_dir, "types.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump([t.name for t in proposals.proposed_types],
                          f,
                          indent=2)
        if proposals.proposed_predicates:
            with open(os.path.join(proposals_dir, "predicates_validated.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump([p.name for p in proposals.proposed_predicates],
                          f,
                          indent=2)
        if proposals.augment_task_code:
            with open(os.path.join(proposals_dir, "augmentor_code.py"),
                      "w",
                      encoding="utf-8") as f:
                f.write(proposals.augment_task_code)
        if proposals.proposed_processes:
            with open(os.path.join(proposals_dir, "processes_code.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump([p.name for p in proposals.proposed_processes],
                          f,
                          indent=2)

        any_retractions = any([
            proposals.retract_type_names,
            proposals.retract_predicate_names,
            proposals.retract_object_augmentor,
            proposals.retract_process_names,
            proposals.retract_option_names,
        ])
        if any_retractions:
            with open(os.path.join(proposals_dir, "retractions.json"),
                      "w",
                      encoding="utf-8") as f:
                json.dump(
                    {
                        "types": sorted(proposals.retract_type_names),
                        "predicates": sorted(
                            proposals.retract_predicate_names),
                        "augmentor": proposals.retract_object_augmentor,
                        "processes": sorted(proposals.retract_process_names),
                        "options": sorted(proposals.retract_option_names),
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
        """Save approach state."""
        save_path = utils.get_approach_save_path_str()
        with open(
                f"{save_path}_{online_learning_cycle}.AgentAbstractionLearning",
                "wb") as f:
            save_dict = {
                "processes":
                self._processes,
                "learned_predicates":
                self._learned_predicates,
                "offline_dataset":
                self._offline_dataset,
                "online_dataset":
                self._online_dataset,
                "online_learning_cycle":
                self._online_learning_cycle,
                "helper_types":
                self._helper_types,
                "augment_task_code":
                self._augment_task_code,
                "agent_proposed_options":
                self._agent_proposed_options,
                "agent_proposed_processes":
                self._agent_proposed_processes,
                "iteration_history":
                self._iteration_history,
                "agent_session_id": (self._agent_session.session_id
                                     if self._agent_session else None),
            }
            pkl.dump(save_dict, f)
            logging.info(f"Saved approach to {save_path}_"
                         f"{online_learning_cycle}.AgentAbstractionLearning")

    def load(self, online_learning_cycle: Optional[int] = None) -> None:
        """Load previously saved approach state."""
        save_path = utils.get_approach_load_path_str()
        with open(
                f"{save_path}_{online_learning_cycle}.AgentAbstractionLearning",
                "rb") as f:
            save_dict = pkl.load(f)

        self._processes = save_dict["processes"]
        self._learned_predicates = save_dict["learned_predicates"]
        self._offline_dataset = save_dict["offline_dataset"]
        self._online_dataset = save_dict["online_dataset"]
        self._online_learning_cycle = save_dict["online_learning_cycle"] + 1
        self._helper_types = save_dict.get("helper_types", set())
        self._augment_task_code = save_dict.get("augment_task_code", "")
        self._agent_proposed_options = save_dict.get("agent_proposed_options",
                                                     set())
        self._agent_proposed_processes = save_dict.get(
            "agent_proposed_processes", set())
        self._iteration_history = save_dict.get("iteration_history", [])
        self._agent_session_id = save_dict.get("agent_session_id")

        # Re-exec augment_task_code to restore the function
        if self._augment_task_code:
            exec_ctx = build_exec_context(self._types,
                                          self._get_current_predicates(),
                                          self._initial_options)
            result, error = exec_code_safely(self._augment_task_code, exec_ctx,
                                             "augment_task")
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


# ------------------------------------------------------------------ #
# Prompt helpers (abstraction-learning specific)
# ------------------------------------------------------------------ #

_SYSTEM_PROMPT = """\
You are an abstraction inventor for a bilevel process planning system. Your \
role is to propose types, predicates, helper objects, processes, and options \
that help a task planner solve planning problems.

## What You Observe

You observe the world ONLY through:
- **Trajectory data**: sequences of states (feature vectors per object) and \
actions
- **Task goals**: symbolic goal descriptions
- **Planning metrics**: success rate, nodes expanded, failure reasons
- **Current abstractions**: the types, predicates, processes, and options \
currently in use

You do NOT have access to environment source code, simulator internals, or \
ground-truth models. You must infer useful abstractions from observed data.

## What You Can Propose

1. **Types**: New object types with named features
2. **Predicates**: Boolean classifiers over states and objects
3. **Helper Objects / Task Augmentation**: Functions that add helper objects \
to tasks (e.g., grid locations, reference frames)
4. **Processes**: Causal processes (exogenous events triggered by conditions)
5. **Options**: Parameterized actions

## Code Conventions

When writing proposal code, the following variables are available in the exec \
context:

### Imports (already available — no need to import)
- `np`, `numpy`, `torch`
- `Box` (from gym.spaces)
- `Type`, `Predicate`, `DerivedPredicate`, `NSPredicate`
- `Object`, `Variable`, `LiftedAtom`, `GroundAtom`
- `ExogenousProcess`, `EndogenousProcess`, `CausalProcess`
- `ParameterizedOption`, `State`, `Task`
- `ConstantDelay`, `DiscreteGaussianDelay`
- `List`, `Set`, `Sequence` (from typing)

### Current abstractions
- Each type `T` is available as `T_type` (e.g., `domino_type`, `robot_type`)
- Each predicate `P` is available by name (e.g., `Fallen`, `Standing`)
- Each predicate classifier is available as `_P_holds` \
(e.g., `_Fallen_holds`)
- Each option `O` is available by name (e.g., `Push`)

### Expected output variables per proposal tool
- `propose_types`: must define `proposed_types` (a list of Type objects)
- `propose_predicates`: must define `proposed_predicates` \
(a list of Predicate objects)
- `propose_object_augmentor`: must define `augment_task(task) -> Task`
- `propose_processes`: must define `proposed_processes` \
(a list of CausalProcess objects)
- `propose_options`: must define `proposed_options` \
(a list of ParameterizedOption objects)

## Key API Reference

### State
```python
state.get(obj, "feature_name")  # get a feature value
state.set(obj, "feature_name", value)  # set a feature value
state.get_objects(some_type)  # get all objects of a type
list(state)  # iterate over all objects
state.copy()  # copy the state
```

### Predicate
```python
pred = Predicate("MyPred", [type1_type, type2_type],
                 lambda state, objects: state.get(objects[0], "feat") > 0.5)
pred.holds(state, [obj1, obj2])  # evaluate
```

### Process (ExogenousProcess)
```python
v1 = Variable("?x", some_type)
v2 = Variable("?y", other_type)
proc = ExogenousProcess(
    name="MyProcess",
    parameters=[v1, v2],
    condition_at_start={LiftedAtom(SomePred, [v1, v2])},
    condition_overall={LiftedAtom(SomePred, [v1, v2])},
    condition_at_end=set(),
    add_effects={LiftedAtom(ResultPred, [v1])},
    delete_effects=set(),
    delay_distribution=ConstantDelay(1),
    strength=torch.tensor([1.0]),
)
```

### Type
```python
my_type = Type("my_type", ["feature1", "feature2"])
```

## Iteration Protocol

At each learning iteration:
1. **Inspect** the trajectory data and planning results using inspection tools
2. **Form hypotheses** about what abstractions are missing or insufficient
3. **Propose** new abstractions using proposal tools
4. **Test** your proposals using testing tools
5. **Refine** based on test results - fix errors and retry

Focus on proposing abstractions that will help the planner solve more tasks. \
Pay attention to:
- States where planning fails - what conditions are missing?
- Patterns in trajectory data that aren't captured by current predicates
- Whether helper objects (like grid positions) could simplify the problem
"""


def build_iteration_message(
        cycle: int,
        num_new_trajs: int,
        num_total_trajs: int,
        task_success_rate: float,
        type_names_with_features: str,
        predicate_signatures: str,
        num_predicates: int,
        process_summaries: str,
        num_processes: int,
        option_names: str,
        num_options: int,
        planning_success: str,
        avg_nodes: str,
        failure_summaries: str,
        previous_iteration_outcomes: str,
        available_tools: Optional[List[Any]] = None) -> str:
    """Build the message sent to the agent at each iteration."""
    tools_section = ""
    if available_tools:
        tool_list = "\n".join(f"  - {t}" for t in available_tools)
        tools_section = f"\nAVAILABLE TOOLS:\n{tool_list}\n"

    return f"""\
== Online Learning Iteration {cycle} ==

TRAJECTORY SUMMARY:
- {num_new_trajs} new trajectories collected this cycle
- {num_total_trajs} total trajectories (offline + online)
- Task success rate: {task_success_rate:.1%}

CURRENT ABSTRACTIONS:
- Types: {type_names_with_features}
- Predicates ({num_predicates}): {predicate_signatures}
- Processes ({num_processes}): {process_summaries}
- Options ({num_options}): {option_names}

PLANNING PERFORMANCE:
{planning_success}
- Avg nodes expanded: {avg_nodes}
- Failures: {failure_summaries}

PREVIOUS ITERATION OUTCOMES:
{previous_iteration_outcomes}
{tools_section}
YOUR TASK:
Inspect the trajectory data and planning results. Propose new or improved \
abstractions that will help the planner solve more tasks. Use the proposal \
tools to register your proposals and the testing tools to validate them.
"""
