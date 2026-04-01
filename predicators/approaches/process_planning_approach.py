"""Bilevel process planning approach."""
import abc
import logging
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.option_model import _OptionModelBase
from predicators.planning import PlanningFailure, PlanningTimeout
from predicators.planning_with_processes import ProcessWorldModel, \
    process_task_plan_grounding, run_task_plan_with_processes_once, \
    sesame_plan_with_processes
from predicators.settings import CFG
from predicators.structs import AbstractProcessPolicy, Action, CausalProcess, \
    EndogenousProcess, GroundAtom, Metrics, Object, ParameterizedOption, \
    Predicate, State, Task, Type, _GroundEndogenousProcess, _Option


class BilevelProcessPlanningApproach(BilevelPlanningApproach):
    """A bilevel planning approach that doesn't use the nsrt world model but
    uses the process world model."""

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
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)
        self._last_option_plan: List[_Option] = []  # used if plan WITH sim

        # Conditionally load VLM components if an abstract policy is used.
        self._vlm = None
        self.base_prompt = ""
        if CFG.process_planning_use_abstract_policy:
            # Set up the vlm and base prompt.
            self._vlm = utils.create_llm_by_name(CFG.llm_model_name)
            # Note: requires a new CFG setting, e.g.,
            # process_planning_vlm_prompt_suffix = "_process"
            prompt_suffix = CFG.process_planning_vlm_prompt_suffix
            root = utils.get_path_to_predicators_root()
            filepath_to_vlm_prompt = (
                root + "/predicators/approaches/"
                "vlm_planning_prompts/no_few_shot_hla_plan"
                f"{prompt_suffix}.txt")
            with open(filepath_to_vlm_prompt, "r", encoding="utf-8") as f:
                self.base_prompt = f.read()

    @abc.abstractmethod
    def _get_current_processes(self) -> Set[CausalProcess]:
        """Get the current set of Processes."""
        raise NotImplementedError("Override me!")

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        # ensure random over successive
        seed = self._seed + self._num_calls
        processes = self._get_current_processes()
        preds = self._get_current_predicates()

        abstract_policy = None
        if CFG.process_planning_use_abstract_policy:
            abstract_policy = self._build_abstract_policy(task)

        # Run task planning only and then greedily sample
        # and execute in the policy.
        if self._plan_without_sim:
            process_plan, atoms_seq, metrics =\
                self._run_task_plan_with_processes(
                    task,
                    processes,
                    preds,
                    timeout,
                    seed,
                    abstract_policy=abstract_policy,
                    max_policy_guided_rollout=CFG.
                    process_planning_max_policy_guided_rollout)
            # pylint: disable=attribute-defined-outside-init
            self._last_process_plan = process_plan
            self._last_atoms_seq = atoms_seq
            # pylint: enable=attribute-defined-outside-init
            policy = utils.process_plan_to_greedy_policy(
                process_plan,
                task.goal,
                self._rng,
                abstract_function=lambda s: utils.abstract(s, preds),
                atoms_seq=atoms_seq)
            logging.debug("Current Task Plan:")
            for process in process_plan:
                logging.debug(process.name)
        else:
            option_plan, process_plan, metrics = \
                self._run_sesame_plan_with_processes(
                    task,
                    processes,
                    preds,
                    timeout,
                    seed,
                    abstract_policy=abstract_policy,
                    max_policy_guided_rollout=CFG.
                    process_planning_max_policy_guided_rollout)
            # pylint: disable=attribute-defined-outside-init
            self._last_option_plan = option_plan
            self._last_process_plan = process_plan
            # pylint: enable=attribute-defined-outside-init
            policy = utils.option_plan_to_policy(option_plan)

        self._save_metrics(metrics, processes, preds)

        def _policy(s: State) -> Action:
            try:
                return policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _run_task_plan_with_processes(
        self, task: Task, processes: Set[CausalProcess], preds: Set[Predicate],
        timeout: int, seed: int, **kwargs: Any
    ) -> Tuple[List[_GroundEndogenousProcess], List[Set[GroundAtom]], Metrics]:
        try:
            plan, atoms_seq, metrics = run_task_plan_with_processes_once(
                task,
                processes,
                preds,
                self._types,
                timeout,
                seed,
                _task_planning_heuristic=self._task_planning_heuristic,
                max_horizon=float(CFG.horizon),
                **kwargs)
        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        return plan, atoms_seq, metrics

    def _run_sesame_plan_with_processes(
        self, task: Task, processes: Set[CausalProcess], preds: Set[Predicate],
        timeout: float, seed: int, **kwargs: Any
    ) -> Tuple[List[_Option], List[_GroundEndogenousProcess], Metrics]:
        """Run full bilevel planning with processes.

        Subclasses may override, e.g. to insert an abstract policy.
        """
        try:
            option_plan, process_plan, metrics = sesame_plan_with_processes(
                task,
                self._option_model,
                processes,
                preds,
                timeout,
                seed,
                max_skeletons_optimized=self._max_skeletons_optimized,
                max_horizon=CFG.horizon,
                **kwargs)
        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        return option_plan, process_plan, metrics

    def _save_metrics(  # type: ignore[override]  # pylint: disable=arguments-renamed
            self, metrics: Metrics, processes: Set[CausalProcess],
            predicates: Set[Predicate]) -> None:
        for metric in [
                "num_samples", "num_skeletons_optimized",
                "num_failures_discovered", "num_nodes_expanded",
                "num_nodes_created", "plan_length", "refinement_time"
        ]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        self._metrics["total_num_processes"] += len(processes)
        self._metrics["total_num_preds"] += len(predicates)
        for metric in [
                "num_samples",
                "num_skeletons_optimized",
        ]:
            self._metrics[f"min_{metric}"] = min(
                metrics[metric], self._metrics[f"min_{metric}"])
            self._metrics[f"max_{metric}"] = max(
                metrics[metric], self._metrics[f"max_{metric}"])

    def _build_abstract_policy(self, task: Task) -> AbstractProcessPolicy:
        """Use a VLM to generate a plan and build a policy from it."""
        # 1. Set up for VLM query.
        init_atoms = utils.abstract(task.init, self._get_current_predicates())
        objects = set(task.init)
        all_processes = self._get_current_processes()
        endogenous_processes = sorted(
            [p for p in all_processes if isinstance(p, EndogenousProcess)])
        vlm_process_plan = self._get_vlm_plan(task, init_atoms, objects,
                                              endogenous_processes)

        # 3. Build the partial policy dictionary by simulating the plan.
        partial_policy_dict: Dict[FrozenSet[GroundAtom],
                                  _GroundEndogenousProcess] = {}
        current_atoms = init_atoms.copy()
        all_ground_processes, _ = process_task_plan_grounding(init_atoms,
                                                              objects,
                                                              all_processes,
                                                              allow_waits=True)
        all_predicates = utils.add_in_auxiliary_predicates(
            self._get_current_predicates())
        derived_predicates = utils.get_derived_predicates(all_predicates)

        # Build indexes for efficient world model execution
        # (do this once outside the loop)
        # pylint: disable=import-outside-toplevel
        from collections import defaultdict

        from predicators.planning_with_processes import \
            _build_exogenous_process_index

        # pylint: enable=import-outside-toplevel

        precondition_to_exogenous_processes = None
        if CFG.build_exogenous_process_index_for_planning:
            precondition_to_exogenous_processes = \
                _build_exogenous_process_index(
                    all_ground_processes)

        # Pre-compute dependencies for incremental derived predicates
        dep_to_derived_preds = defaultdict(list)
        for der_pred in derived_predicates:
            for aux_pred in (der_pred.auxiliary_predicates or set()):
                dep_to_derived_preds[aux_pred].append(der_pred)

        for ground_process in vlm_process_plan:
            if not ground_process.condition_at_start.issubset(current_atoms):
                logging.warning(f"VLM plan deviates, precondition not met for "
                                f"{ground_process.name_and_objects_str()}")
                break

            frozen_atoms = frozenset(current_atoms)
            partial_policy_dict[frozen_atoms] = ground_process

            # Simulate the step to get the next state with proper indexing.
            world_model = ProcessWorldModel(
                ground_processes=all_ground_processes,
                state=current_atoms.copy(),
                state_history=[],
                action_history=[],
                scheduled_events={},
                t=0,
                derived_predicates=derived_predicates,
                objects=objects,
                precondition_to_exogenous_processes=
                precondition_to_exogenous_processes,
                dep_to_derived_preds=dep_to_derived_preds)

            world_model.big_step(ground_process)
            current_atoms = world_model.state

        # 4. Create and return the abstract policy.
        abstract_policy = lambda atoms, _1, _2: partial_policy_dict.get(
            frozenset(atoms), None)

        return abstract_policy

    def _get_vlm_plan(
        self, task: Task, init_atoms: Set[GroundAtom], objects: Set[Object],
        endogenous_processes: List[EndogenousProcess]
    ) -> List[_GroundEndogenousProcess]:

        # 2. Query VLM for a process plan.
        processes_str = "\n".join(str(p) for p in endogenous_processes)
        objects_list = sorted(list(objects))
        objects_str = "\n".join(str(obj) for obj in objects_list)
        goal_str = "\n".join(str(g) for g in sorted(task.goal))
        type_hierarchy_str = utils.create_pddl_types_str(self._types)
        init_state_str = "\n".join(map(str, sorted(init_atoms)))

        prompt = self.base_prompt.format(processes=processes_str,
                                         typed_objects=objects_str,
                                         type_hierarchy=type_hierarchy_str,
                                         init_state_str=init_state_str,
                                         goal_str=goal_str)

        try:
            assert self._vlm is not None
            vlm_output = self._vlm.sample_completions(
                prompt,
                imgs=None,  # No images for process planning.
                temperature=CFG.vlm_temperature,
                seed=CFG.seed,
                num_completions=1)
            plan_prediction_txt = vlm_output[0]
            start_index = plan_prediction_txt.index("Plan:\n") + len("Plan:\n")
            parsable_plan_prediction = plan_prediction_txt[start_index:]
        except (ValueError, IndexError, AssertionError) as e:
            logging.warning(f"VLM output parsing failed, returning trivial "
                            f"policy. Reason: {e}")
            # Return an empty plan on parsing failure
            vlm_process_plan = []

        # Note: this requires a new utility function,
        # `parse_model_output_into_process_plan`, which should be analogous
        # to `parse_model_output_into_option_plan`.
        try:
            parse_fn = (
                utils  # type: ignore[attr-defined]  # pylint: disable=no-member
                .parse_model_output_into_process_plan)
            parsed_process_plan = parse_fn(parsable_plan_prediction,
                                           objects_list, self._types,
                                           endogenous_processes)
            vlm_process_plan = [
                p.ground(objs) for p, objs in parsed_process_plan
            ]
        except Exception as e:  # pylint: disable=broad-except
            logging.warning("Failed to parse/ground VLM process plan:"
                            f" {e}")
            vlm_process_plan = []

        return vlm_process_plan
