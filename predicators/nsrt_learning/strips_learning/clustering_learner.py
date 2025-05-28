"""Algorithms for STRIPS learning that rely on clustering to obtain effects."""

import abc
import functools
import itertools
import logging
import multiprocessing as mp
import re
from collections import defaultdict
from pprint import pformat
from typing import Dict, FrozenSet, Iterator, List, Optional, Set, Tuple, cast

from predicators import utils
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, Datastore, DummyOption, LiftedAtom, \
    ParameterizedOption, Predicate, Segment, STRIPSOperator, VarToObjSub, \
    ExogenousProcess, EndogenousProcess
from predicators.planning import task_plan_grounding, PlanningFailure, \
    PlanningTimeout
from predicators.planning_with_processes import \
    task_plan as task_plan_with_processes


class ClusteringSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a clustering-based STRIPS learner."""

    def _learn(self) -> List[PNAD]:
        segments = [seg for segs in self._segmented_trajs for seg in segs]
        # Cluster the segments according to common option and effects.
        pnads: List[PNAD] = []
        for segment in segments:
            if segment.has_option():
                segment_option = segment.get_option()
                segment_param_option = segment_option.parent
                segment_option_objs = tuple(segment_option.objects)
            else:
                segment_param_option = DummyOption.parent
                segment_option_objs = tuple()
            for pnad in pnads:
                # Try to unify this transition with existing effects.
                # Note that both add and delete effects must unify,
                # and also the objects that are arguments to the options.
                (pnad_param_option, pnad_option_vars) = pnad.option_spec
                if self.get_name() not in [
                        "cluster_and_llm_select",
                        "cluster_and_search_process_learner",
                        "cluster_and_inverse_planning"
                ]:
                    preconds1 = frozenset()  # no preconditions
                    preconds2 = frozenset()  # no preconditions
                else:
                    preconds1 = frozenset(segment.init_atoms)
                    obj_to_var = {
                        v: k
                        for k, v in pnad.datastore[0][1].items()
                    }
                    preconds2 = frozenset({
                        atom.lift(obj_to_var)
                        for atom in pnad.datastore[0][0].init_atoms
                    })
                suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                    preconds1, preconds2, frozenset(segment.add_effects),
                    frozenset(pnad.op.add_effects),
                    frozenset(segment.delete_effects),
                    frozenset(pnad.op.delete_effects), segment_param_option,
                    pnad_param_option, segment_option_objs,
                    tuple(pnad_option_vars))
                sub = cast(VarToObjSub,
                           {v: o
                            for o, v in ent_to_ent_sub.items()})
                if suc:
                    # Add to this PNAD.
                    assert set(sub.keys()) == set(pnad.op.parameters)
                    pnad.add_to_datastore((segment, sub))
                    break
            else:
                # Otherwise, create a new PNAD.
                objects = {o for atom in segment.add_effects |
                           segment.delete_effects for o in atom.objects} | \
                          set(segment_option_objs)

                if self.get_name() in [
                        "cluster_and_llm_select",
                        "cluster_and_search_process_learner",
                        "cluster_and_inverse_planning"
                ]:
                    # With cluster_and_llm_select, the param may include
                    # anything in the init atoms of the segment.
                    objects |= {
                        o
                        for atom in segment.init_atoms for o in atom.objects
                    }

                objects_lst = sorted(objects)
                params = utils.create_new_variables(
                    [o.type for o in objects_lst])
                preconds: Set[LiftedAtom] = set()  # will be learned later
                obj_to_var = dict(zip(objects_lst, params))
                var_to_obj = dict(zip(params, objects_lst))
                add_effects = {
                    atom.lift(obj_to_var)
                    for atom in segment.add_effects
                }
                delete_effects = {
                    atom.lift(obj_to_var)
                    for atom in segment.delete_effects
                }
                ignore_effects: Set[Predicate] = set()  # will be learned later
                op = STRIPSOperator(f"Op{len(pnads)}", params, preconds,
                                    add_effects, delete_effects,
                                    ignore_effects)
                datastore = [(segment, var_to_obj)]
                option_vars = [obj_to_var[o] for o in segment_option_objs]
                option_spec = (segment_param_option, option_vars)
                pnads.append(PNAD(op, datastore, option_spec))

        if self.get_name() in ["cluster_and_search_process_learner"]:
            # Do this extra step for this learner
            initial_segmenter_method = CFG.segmenter
            CFG.segmenter = "atom_changes"
            self._atom_change_segmented_trajs = [
                segment_trajectory(traj, self._predicates, verbose=False)
                for traj in self._trajectories
            ]
            CFG.segmenter = initial_segmenter_method
        # Learn the preconditions of the operators in the PNADs. This part
        # is flexible; subclasses choose how to implement it.
        pnads = self._learn_pnad_preconditions(pnads)

        # Handle optional postprocessing to learn ignore effects.
        pnads = self._postprocessing_learn_ignore_effects(pnads)

        # Log and return the PNADs.
        if self._verbose:
            logging.info("Learned operators (before option learning):")
            for pnad in pnads:
                logging.info(pnad)
        return pnads

    @abc.abstractmethod
    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Subclass-specific algorithm for learning PNAD preconditions.

        Returns a list of new PNADs. Should NOT modify the given PNADs.
        """
        raise NotImplementedError("Override me!")

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        """Optionally postprocess to learn ignore effects."""
        _ = self  # unused, but may be used in subclasses
        return pnads


class ClusterAndIntersectSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via
    intersection."""

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        new_pnads = []
        for pnad in pnads:
            if CFG.cluster_and_intersect_soft_intersection_for_preconditions:
                preconditions = \
                    self._induce_preconditions_via_soft_intersection(pnad)
            else:
                preconditions = self._induce_preconditions_via_intersection(
                    pnad)
            # Since we are taking an intersection, we're guaranteed that the
            # datastore can't change, so we can safely use pnad.datastore here.
            new_pnads.append(
                PNAD(pnad.op.copy_with(preconditions=preconditions),
                     pnad.datastore, pnad.option_spec))
        return new_pnads

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect"

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        """Prune PNADs whose datastores are too small.

        Specifically, keep PNADs that have at least
        CFG.cluster_and_intersect_min_datastore_fraction fraction of the
        segments produced by the option in their NSRT.
        """
        if not CFG.cluster_and_intersect_prune_low_data_pnads:
            return pnads
        option_to_dataset_size: Dict[ParameterizedOption,
                                     int] = defaultdict(int)
        for pnad in pnads:
            option = pnad.option_spec[0]
            option_to_dataset_size[option] += len(pnad.datastore)
        ret_pnads: List[PNAD] = []
        for pnad in pnads:
            option = pnad.option_spec[0]
            fraction = len(pnad.datastore) / option_to_dataset_size[option]
            if fraction >= CFG.cluster_and_intersect_min_datastore_fraction:
                ret_pnads.append(pnad)
        return ret_pnads


class ClusterAndLLMSelectSTRIPSLearner(ClusteringSTRIPSLearner):
    """Learn preconditions via LLM selection.

    Note: The current prompt are tailored for exogenous processes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._llm = utils.create_llm_by_name(CFG.llm_model_name)
        prompt_file = utils.get_path_to_predicators_root() + \
            "/predicators/nsrt_learning/strips_learning/" + \
            "llm_op_learning_prompts/condition_selection.prompt"
        with open(prompt_file, "r") as f:
            self.base_prompt = f.read()
        from predicators.approaches.pp_online_predicate_invention_approach import \
            get_false_positive_process_states
        self._get_false_positive_process_states = get_false_positive_process_states

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_llm_select"

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Assume there is one segment per PNAD We can either do lifting first
        and selection second, or the other way around.

        If we have multiple segments per PNAD, lifting requires us to
        find a subset of atoms that unifies the segments. We'd have to
        do this if we want to learn a single condition. But we could
        also learn more than one.
        """
        # Add var_to_obj for objects in the init state of the segment
        new_pnads = []
        for pnad in pnads:
            # Removing this assumption because we're now making sure that
            # all the init_atoms in the PNAD are the same up to unification.
            # assert len(pnad.datastore) == 1
            seg, var_to_obj = pnad.datastore[0]
            existing_objs = set(var_to_obj.values())
            # Get the init atoms of the segment
            init_atoms = seg.init_atoms
            # Get the objects in the init atoms
            additional_objects = {
                o
                for atom in init_atoms for o in atom.objects
                if o not in existing_objs
            }
            # Create a new var_to_obj mapping for the objects
            objects_lst = sorted(additional_objects)
            params = utils.create_new_variables([o.type for o in objects_lst],
                                                existing_vars=list(var_to_obj))
            var_to_obj.update(dict(zip(params, objects_lst)))
            new_pnads.append(
                PNAD(pnad.op, [(seg, var_to_obj)],
                     pnad.option_spec))  # dummy option

        seperate_llm_query_per_pnad = True
        effect_and_conditions = ""
        proposed_conditions: List[str] = []
        for i, pnad in enumerate(new_pnads):
            if seperate_llm_query_per_pnad:
                effect_and_conditions += f"Process 0:\n"
            else:
                effect_and_conditions += f"Process {i}:\n"
            add_effects = pnad.op.add_effects
            delete_effects = pnad.op.delete_effects
            effect_and_conditions += "Add effects: ("
            if add_effects:
                effect_and_conditions += "and " + " ".join(f"({str(atom)})" for\
                                                           atom in add_effects)
            effect_and_conditions += ")\n"
            effect_and_conditions += "Delete effects: ("
            if delete_effects:
                effect_and_conditions += "and " +  " ".join(f"({str(atom)})" \
                                                        for atom in delete_effects)
            effect_and_conditions += ")\n"
            segment_init_atoms = pnad.datastore[0][0].init_atoms
            segment_var_to_obj = pnad.datastore[0][1]
            obj_to_var = {v: k for k, v in segment_var_to_obj.items()}
            conditions_to_choose_from = pformat(
                {a.lift(obj_to_var)
                 for a in segment_init_atoms})
            effect_and_conditions += "Conditions to choose from:\n" +\
                conditions_to_choose_from + "\n\n"

            if seperate_llm_query_per_pnad:
                prompt = self.base_prompt.format(
                    EFFECTS_AND_CONDITIONS=effect_and_conditions)
                proposals = self._llm.sample_completions(
                    prompt, None, 0.0, CFG.seed)[0]
                pattern = r'```\n(.*?)\n```'
                matches = re.findall(pattern, proposals, re.DOTALL)
                proposed_conditions.append(matches[0])
                effect_and_conditions = ""

        if not seperate_llm_query_per_pnad:
            prompt = self.base_prompt.format(
                EFFECTS_AND_CONDITIONS=effect_and_conditions)
            proposals = self._llm.sample_completions(prompt, None, 0.0,
                                                     CFG.seed)[0]
            pattern = r'```\n(.*?)\n```'
            matches = re.findall(pattern, proposals, re.DOTALL)
            proposed_conditions = matches[0].split("\n\n")

        def atom_in_llm_selection(
                atom: LiftedAtom,
                conditions: List[Tuple[str, List[Tuple[str, str]]]]) -> bool:
            for condition in conditions:
                atom_name = condition[0]
                atom_variables = condition[1]
                if atom.predicate.name == atom_name and \
                        all([var_type[0] == var.name for (var_type, var) in
                            zip(atom_variables, atom.variables)]):
                    return True
            return False

        # Assumes the same number of PNADs and response chunks
        assert len(new_pnads) == len(proposed_conditions)
        final_pnads: List[PNAD] = []
        for proposed_condition, corresponding_pnad in zip(
                proposed_conditions, new_pnads):
            # Get the effect atoms
            # Get the condition atoms
            lines = proposed_condition.split("\n")
            # add_effects = self.parse_effects_or_conditions(lines[0])
            # delete_effects = self.parse_effects_or_conditions(lines[1])
            conditions = self.parse_effects_or_conditions(lines[2])

            segment_init_atoms = corresponding_pnad.datastore[0][0].init_atoms
            segment_var_to_obj = corresponding_pnad.datastore[0][1]
            obj_to_var = {v: k for k, v in segment_var_to_obj.items()}
            conditions_to_choose_from = {
                a.lift(obj_to_var)
                for a in segment_init_atoms
            }
            new_conditions = set(atom for atom in conditions_to_choose_from
                                 if atom_in_llm_selection(atom, conditions))
            add_eff = corresponding_pnad.op.add_effects
            del_eff = corresponding_pnad.op.delete_effects
            # the variable might also just in the effects
            new_parameters = set(var
                                 for atom in new_conditions | add_eff | del_eff
                                 for var in atom.variables)
            # Only append if it's unique
            for final_pnad in final_pnads:
                suc, _ = utils.unify_preconds_effects_options(
                    frozenset(new_conditions),
                    frozenset(final_pnad.op.preconditions),
                    frozenset(corresponding_pnad.op.add_effects),
                    frozenset(final_pnad.op.add_effects),
                    frozenset(corresponding_pnad.op.delete_effects),
                    frozenset(final_pnad.op.delete_effects),
                    corresponding_pnad.option_spec[0],
                    final_pnad.option_spec[0],
                    tuple(corresponding_pnad.option_spec[1]),
                    tuple(final_pnad.option_spec[1]),
                )
                if suc:
                    break
            else:
                # We have a new process!
                # Create a new PNAD with the new parameters and conditions
                # and add it to the final list
                pnad = PNAD(
                    corresponding_pnad.op.copy_with(
                        parameters=new_parameters,
                        preconditions=new_conditions),
                    corresponding_pnad.datastore,
                    corresponding_pnad.option_spec)
                final_pnads.append(pnad)

                if CFG.process_learner_check_false_positives:
                    # Go through the trajectories and check if this process
                    # leads to false positive effect predications.
                    false_positive_process_state = \
                        self._get_false_positive_process_states(
                            self._trajectories,
                            self._predicates,
                            [pnad.make_exogenous_process()])

                    for _, states in false_positive_process_state.items():
                        if len(states) > 0:
                            # initial_segmenter_method = CFG.segmenter
                            # CFG.segmenter = "atom_changes"
                            # segments = [segment_trajectory(traj, self._predicates) for traj in self._trajectories]
                            # CFG.segmenter = initial_segmenter_method
                            breakpoint()
        return final_pnads

    def parse_effects_or_conditions(
            self, line: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """Parse a line containing effects or conditions into a list of tuples.
        For example, when given: 'Conditions: (and (FaucetOn(?x1:faucet))
        (JugUnderFaucet(?x2:jug, ?x1:faucet)))'.

        Each returned tuple has:
        - An atom name (e.g., "JugFilled")
        - A list of (variable_name, type_name) pairs
        (e.g., [("?x0", "jug"), ("?x1", "faucet")]).

        Example Return:
        [
            ("FaucetOn", [("?x1", "faucet")]),
            ("JugUnderFaucet", [("?x2", "jug"), ("?x1", "faucet")])
        ]
        """

        # Remove the top-level (and ...) if present.
        # This way, we won't accidentally capture "and" as an atom.
        line = re.sub(r"\(\s*and\s+", "(", line)

        # Match an atom name and the entire content inside its parentheses.
        pattern = r"\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\)"
        atom_matches = re.findall(pattern, line)

        var_type_pattern = r"(\?[a-zA-Z0-9]+):([a-zA-Z0-9_]+)"
        parsed_atoms: List[Tuple[str, List[Tuple[str, str]]]] = []

        for atom_name, vars_str in atom_matches:
            # Find all variable:type pairs in the string
            var_type_pairs = re.findall(var_type_pattern, vars_str)
            parsed_atoms.append((atom_name, var_type_pairs))

        return parsed_atoms


class ClusterAndSearchProcessLearner(ClusteringSTRIPSLearner):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from predicators.approaches.pp_online_predicate_invention_approach import \
            get_false_positive_process_states_from_segmented_trajs
        self._get_false_positive_process_states_from_segmented_trajs = \
            get_false_positive_process_states_from_segmented_trajs
        self._atom_change_segmented_trajs: List[List[Segment]] = []

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_search_process_learner"

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        # Check if parallelization is enabled and beneficial
        use_parallel = (CFG.cluster_and_search_process_learner_use_parallel
                        and len(pnads) > 1 and mp.cpu_count() > 1)

        if use_parallel:
            return self._learn_pnad_preconditions_parallel(pnads)
        else:
            return self._learn_pnad_preconditions_sequential(pnads)

    def _learn_pnad_preconditions_parallel(self,
                                           pnads: List[PNAD]) -> List[PNAD]:
        """Parallelized version of precondition learning."""

        # Determine number of workers
        if CFG.cluster_search_max_workers == -1:
            max_workers = mp.cpu_count()
        else:
            max_workers = CFG.cluster_search_max_workers
        num_workers = min(max_workers, len(pnads))

        logging.info(
            f"Running parallel search with {num_workers} workers for {len(pnads)} PNADs"
        )

        # Phase 1: Run searches in parallel
        search_results = []
        try:
            # Use spawn method to avoid issues with copying complex objects
            ctx = mp.get_context('spawn')
            with ctx.Pool(num_workers) as pool:
                # Create partial function with shared data
                search_worker = functools.partial(
                    self._run_search_worker,
                    atom_change_segmented_trajs=self.
                    _atom_change_segmented_trajs)

                # Run searches in parallel
                search_results = pool.map(search_worker, pnads)

        except Exception as e:
            logging.warning(
                f"Parallel processing failed: {e}. Falling back to sequential."
            )
            return self._learn_pnad_preconditions_sequential(pnads)

        # Phase 2: Process results and check uniqueness sequentially
        final_pnads: List[PNAD] = []

        for pnad, precon in zip(pnads, search_results):
            if precon is None:
                logging.warning(
                    f"Search failed for PNAD {pnad.op.name}, skipping.")
                continue

            add_eff = pnad.op.add_effects
            del_eff = pnad.op.delete_effects
            new_params = set(var for atom in precon | add_eff | del_eff
                             for var in atom.variables)

            # Check uniqueness against existing final_pnads
            if self._is_unique_pnad(precon, pnad, final_pnads):
                new_pnad = PNAD(
                    pnad.op.copy_with(preconditions=precon,
                                      parameters=new_params), pnad.datastore,
                    pnad.option_spec)
                final_pnads.append(new_pnad)

        return final_pnads

    def _learn_pnad_preconditions_sequential(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        """Sequential version (original implementation)."""
        final_pnads: List[PNAD] = []

        for pnad in pnads:
            precon = self._run_search(pnad)
            add_eff = pnad.op.add_effects
            del_eff = pnad.op.delete_effects
            new_params = set(var for atom in precon | add_eff | del_eff
                             for var in atom.variables)

            # Check uniqueness
            if self._is_unique_pnad(precon, pnad, final_pnads):
                new_pnad = PNAD(
                    pnad.op.copy_with(preconditions=precon,
                                      parameters=new_params), pnad.datastore,
                    pnad.option_spec)
                final_pnads.append(new_pnad)

        return final_pnads

    def _is_unique_pnad(self, precon: FrozenSet[LiftedAtom], pnad: PNAD,
                        final_pnads: List[PNAD]) -> bool:
        """Check if a PNAD with given preconditions is unique."""
        for final_pnad in final_pnads:
            # Quick size checks first for efficiency
            if (len(precon) != len(final_pnad.op.preconditions) or
                    len(pnad.op.add_effects) != len(final_pnad.op.add_effects)
                    or len(pnad.op.delete_effects) != len(
                        final_pnad.op.delete_effects)):
                continue

            suc, _ = utils.unify_preconds_effects_options(
                frozenset(precon),
                frozenset(final_pnad.op.preconditions),
                frozenset(pnad.op.add_effects),
                frozenset(final_pnad.op.add_effects),
                frozenset(pnad.op.delete_effects),
                frozenset(final_pnad.op.delete_effects),
                pnad.option_spec[0],
                final_pnad.option_spec[0],
                tuple(pnad.option_spec[1]),
                tuple(final_pnad.option_spec[1]),
            )
            if suc:
                return False
        return True

    @staticmethod
    def _run_search_worker(
        pnad: PNAD, atom_change_segmented_trajs: List[List[Segment]]
    ) -> Optional[FrozenSet[LiftedAtom]]:
        """Worker function for parallel search execution."""
        try:
            # Import here to avoid pickling issues
            import functools

            from predicators import utils
            from predicators.approaches.pp_online_predicate_invention_approach import \
                get_false_positive_process_states_from_segmented_trajs
            from predicators.settings import CFG

            # Recreate the search function
            init_ground_atoms = pnad.datastore[0][0].init_atoms
            var_to_obj = pnad.datastore[0][1]
            obj_to_var = {v: k for k, v in var_to_obj.items()}
            initial_state: FrozenSet[LiftedAtom] = frozenset(
                atom.lift(obj_to_var) for atom in init_ground_atoms)

            # Define score function for this worker
            def score_preconditions(
                    preconditions: FrozenSet[LiftedAtom]) -> float:
                exogenous_process = pnad.op.copy_with(
                    preconditions=preconditions).make_exogenous_process()
                false_positive_process_state = \
                    get_false_positive_process_states_from_segmented_trajs(
                        atom_change_segmented_trajs, [exogenous_process])
                num_false_positives = sum(
                    len(states)
                    for states in false_positive_process_state.values())

                complexity_penalty = CFG.grammar_search_pred_complexity_weight * \
                                    len(preconditions)
                return num_false_positives + complexity_penalty

            # Define successor function
            def get_preconditions_successors(
                    preconditions: FrozenSet[LiftedAtom]):
                preconditions_sorted = sorted(preconditions)
                for i in range(len(preconditions_sorted)):
                    successor = preconditions_sorted[:i] + preconditions_sorted[
                        i + 1:]
                    yield i, frozenset(successor), 1.0

            # Run the search
            path, _ = utils.run_gbfs(initial_state, lambda s: False,
                                     get_preconditions_successors,
                                     score_preconditions)

            return path[-1]

        except Exception as e:
            logging.warning(
                f"Search worker failed for PNAD {pnad.op.name}: {e}")
            return None

    def _run_search(self, pnad: PNAD) -> FrozenSet[LiftedAtom]:

        init_ground_atoms = pnad.datastore[0][0].init_atoms
        var_to_obj = pnad.datastore[0][1]
        obj_to_var = {v: k for k, v in var_to_obj.items()}
        initial_state: FrozenSet[LiftedAtom] = frozenset(
            atom.lift(obj_to_var) for atom in init_ground_atoms)
        exogenous_process = pnad.make_exogenous_process()
        score_func = functools.partial(self._score_preconditions, 
                                       exogenous_process)

        path, _ = utils.run_gbfs(initial_state, lambda s: False,
                                 self._get_preconditions_successors,
                                 score_func)

        return_precon = path[-1]
        # logging.debug(f"Search finished. Selected:")
        # score_func(return_precon)
        return return_precon

    def _score_preconditions(self, exogenous_process: ExogenousProcess,
                             preconditions: FrozenSet[LiftedAtom]) -> float:
        exogenous_process.condition_at_start = set(preconditions)
        exogenous_process.condition_overall = set(preconditions)
        false_positive_process_state =\
            self._get_false_positive_process_states_from_segmented_trajs(
                self._atom_change_segmented_trajs, [exogenous_process])
        num_false_positives = 0
        for _, states in false_positive_process_state.items():
            num_false_positives += len(states)
            # logging.debug(states)

        complexity_penalty = CFG.grammar_search_pred_complexity_weight *\
                                    len(preconditions)
        cost = num_false_positives + complexity_penalty
        # logging.debug(f"Score for precon {set(preconditions)}: {cost:.4f}")
        return cost

    def _get_preconditions_successors(
        self, preconditions: FrozenSet[LiftedAtom]
    ) -> Iterator[Tuple[int, FrozenSet[LiftedAtom], float]]:
        """The successors remove each atom in the preconditions."""
        preconditions_sorted = sorted(preconditions)
        for i in range(len(preconditions_sorted)):
            successor = preconditions_sorted[:i] + preconditions_sorted[i + 1:]
            yield i, frozenset(successor), 1.0

class ClusterAndInversePlanningProcessLearner(ClusteringSTRIPSLearner):
    def __init__(self, *args, 
                 endogenous_processes: Set[EndogenousProcess],
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._endogenous_processes = endogenous_processes

        from predicators.predicate_search_score_functions import \
            _ExpectedNodesScoreFunction
        self._get_optimality_prob =\
            _ExpectedNodesScoreFunction._get_refinement_prob

        from predicators.approaches.pp_online_predicate_invention_approach import \
            get_false_positive_process_states_from_segmented_trajs
        self._get_false_positive_process_states_from_segmented_trajs = \
            get_false_positive_process_states_from_segmented_trajs
        
        self._atom_change_segmented_trajs: List[List[Segment]] = []
        self._option_change_segmented_trajs: List[List[Segment]] = []

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_inverse_planning"

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Find the set of PNADs (with corresponding processes) that allows the
        agent make similar plans as the demonstrated/successful plans."""

        # --- Existing exogenous processes ---
        exogenous_process = [pnad.make_exogenous_process() for pnad in pnads]

        # Get the segmented trajectories for scoring the processes.
        initial_segmenter_method = CFG.segmenter
        CFG.segmenter = "atom_changes"
        self._atom_change_segmented_trajs = [
            segment_trajectory(traj, self._predicates, verbose=False)
            for traj in self._trajectories
        ]
        CFG.segmenter = "option_changes"
        self._option_change_segmented_trajs = [
            segment_trajectory(traj, self._predicates, verbose=False)
            for traj in self._trajectories
        ]
        CFG.segmenter = initial_segmenter_method

        # --- Get the candidate preconditions ---
        # First option. Candidates are all possible subsets.
        conditions_at_start = []
        for pnad in pnads:
            init_ground_atoms = pnad.datastore[0][0].init_atoms
            var_to_obj = pnad.datastore[0][1]
            obj_to_var = {v: k for k, v in var_to_obj.items()}
            init_lift_atoms = set(atom.lift(obj_to_var) for atom in 
                                    init_ground_atoms)
            if CFG.cluster_and_inverse_planning_candidates == "all":
                # 4 PNADS, with 7, 6, 7, 8 init atoms, possible combinations are
                # - 2^7 * 2^6 * 2^7 * 2^8 = 2^28 = 268,435,456
                # - 2^10 * 2^10 * 2^10 * 2^10 = 2^40 = 1,099,511,627,776
                # Get the initial conditions of the PNAD
                conditions_at_start.append(utils.all_subsets(init_lift_atoms))
            elif CFG.cluster_and_inverse_planning_candidates == "top_consistent":
                conditions_at_start.append(self.get_top_consistent_conditions(
                    init_lift_atoms, pnad))
            else:
                raise NotImplementedError

        # --- Search for the best combination of preconditions ---
        best_score = -float("inf")
        best_conditions = []
        breakpoint()
        # Score all combinations of preconditions
        for combination in itertools.product(*conditions_at_start):
            # Set the conditions for each process
            for process, conditions in zip(exogenous_process, combination):
                process.condition_at_start = conditions
                process.condition_overall = conditions
            
            # Score this set of processes
            score = self.compute_processes_score(set(exogenous_process))
            if score > best_score:
                best_score = score
                best_conditions = combination

        # --- Create new PNADs with the best conditions ---
        final_pnads: List[PNAD] = []
        for pnad, conditions in zip(pnads, best_conditions): 
            # Check if this PNAD is unique
            for final_pnad in final_pnads:
                suc, _ = utils.unify_preconds_effects_options(
                    frozenset(conditions),
                    frozenset(final_pnad.op.preconditions),
                    frozenset(pnad.op.add_effects),
                    frozenset(final_pnad.op.add_effects),
                    frozenset(pnad.op.delete_effects),
                    frozenset(final_pnad.op.delete_effects),
                    pnad.option_spec[0],
                    final_pnad.option_spec[0],
                    tuple(pnad.option_spec[1]),
                    tuple(final_pnad.option_spec[1]),
                )
                if suc:
                    # TODO: merge datastores if they are the same
                    break
            else:
                # If we reach here, it means the PNAD is unique
                # and we can add it to the final list
                new_pnad = PNAD(
                    pnad.op.copy_with(preconditions=conditions),
                    pnad.datastore, pnad.option_spec)
                final_pnads.append(new_pnad)
        return final_pnads
    
    def get_top_consistent_conditions(self, initial_atom: Set[LiftedAtom],
                                      pnad: PNAD) -> Iterator[Set[LiftedAtom]]:
        """Get the top consistent conditions for a PNAD.
        """
        # TODO: implement the retrieval of the top n scoring ones.
        exogenous_process = pnad.make_exogenous_process()
        # candidates = []

        for atoms in utils.all_subsets(initial_atom):
            exogenous_process.condition_at_start = atoms
            exogenous_process.condition_overall = atoms

            # Check if the process is consistent with the trajectories
            false_positive_process_state = \
                self._get_false_positive_process_states_from_segmented_trajs(
                    self._atom_change_segmented_trajs, [exogenous_process])
            num_false_positives = 0
            for _, states in false_positive_process_state.items():
                num_false_positives += len(states)
            logging.debug(f"Conditions: {atoms}, FP: {num_false_positives}")
            if num_false_positives <=\
                CFG.cluster_and_inverse_planning_top_consistent_max_cost:
                # candidates.append(atoms)
                yield atoms
        # return candidates


    def compute_processes_score(self, exogenous_processes: Set[ExogenousProcess]
                                ) -> float:
        """Score the PNAD based on how well it allows the agent to make plans."""
        # TODO: also incorporate expected number of nodes expanded
        score = 0.0
        for i, traj in enumerate(self._trajectories):
            if not traj.is_demo:
                continue
            demo_atoms_sequence = utils.segment_trajectory_to_atoms_sequence(
                self._option_change_segmented_trajs[i])
            init_atoms = self._option_change_segmented_trajs[i][0].init_atoms
            objects = set(traj.states[0])
            goal = self._train_tasks[traj.train_task_idx].goal
            ground_processes, reachable_atoms = task_plan_grounding(
                init_atoms,
                objects,
                exogenous_processes | self._endogenous_processes,
                allow_noops=True,
                compute_reachable_atoms=False)
            heuristics = utils.create_task_planning_heuristic(
                CFG.sesame_task_planning_heuristic, init_atoms, goal,
                ground_processes,
                self._predicates, objects)
            generator = task_plan_with_processes(
                init_atoms,
                goal,
                ground_processes,
                reachable_atoms,
                heuristics,
                CFG.seed,
                CFG.grammar_search_task_planning_timeout,
                # max_skeletons_optimized=CFG.sesame_max_skeletons_optimized,
                max_skeletons_optimized=1,
                use_visited_state_set=True)
            
            optimality_prob = 0.0
            try:
                for idx, (_, plan_atoms_sequence, 
                          metrics) in enumerate(generator):
                    optimality_prob = self._get_optimality_prob(
                        demo_atoms_sequence, plan_atoms_sequence)
            except (PlanningTimeout, PlanningFailure):
                pass
            score += optimality_prob

        return score

class ClusterAndSearchSTRIPSLearner(ClusteringSTRIPSLearner):
    """A clustering STRIPS learner that learns preconditions via search,
    following the LOFT algorithm: https://arxiv.org/abs/2103.00589."""

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        new_pnads = []
        for i, pnad in enumerate(pnads):
            positive_data = pnad.datastore
            # Construct negative data by merging the datastores of all
            # other PNADs that have the same option.
            negative_data = []
            for j, other_pnad in enumerate(pnads):
                if i == j:
                    continue
                if pnad.option_spec[0] != other_pnad.option_spec[0]:
                    continue
                negative_data.extend(other_pnad.datastore)
            # Run the top-level search to find sets of precondition sets. This
            # also produces datastores, letting us avoid making a potentially
            # expensive call to recompute_datastores_from_segments().
            all_preconditions_to_datastores = self._run_outer_search(
                pnad, positive_data, negative_data)
            for j, preconditions in enumerate(all_preconditions_to_datastores):
                datastore = all_preconditions_to_datastores[preconditions]
                new_pnads.append(
                    PNAD(
                        pnad.op.copy_with(name=f"{pnad.op.name}-{j}",
                                          preconditions=preconditions),
                        datastore, pnad.option_spec))
        return new_pnads

    def _run_outer_search(
            self, pnad: PNAD, positive_data: Datastore,
            negative_data: Datastore
    ) -> Dict[FrozenSet[LiftedAtom], Datastore]:
        """Run outer-level search to find a set of precondition sets and
        associated datastores.

        Each precondition set will produce one operator.
        """
        all_preconditions_to_datastores = {}
        # We'll remove positives as they get covered.
        remaining_positives = list(positive_data)
        while remaining_positives:
            new_preconditions = self._run_inner_search(pnad,
                                                       remaining_positives,
                                                       negative_data)
            # Compute the datastore and update the remaining positives.
            datastore = []
            new_remaining_positives = []
            for seg, var_to_obj in remaining_positives:
                ground_pre = {a.ground(var_to_obj) for a in new_preconditions}
                if not ground_pre.issubset(seg.init_atoms):
                    # If the preconditions ground with this substitution don't
                    # hold in this segment's init_atoms, this segment has yet
                    # to be covered, so we keep it in the positives.
                    new_remaining_positives.append((seg, var_to_obj))
                else:
                    # Otherwise, we can add this segment to the datastore and
                    # also move it to negative_data, for any future
                    # preconditions that get learned.
                    datastore.append((seg, var_to_obj))
                    negative_data.append((seg, var_to_obj))
            # Special case: if the datastore is empty, that means these
            # new_preconditions don't cover any positives, so the search
            # failed to find preconditions that have a better score than inf.
            # Therefore we give up, without including these new_preconditions
            # into all_preconditions_to_datastores.
            if len(datastore) == 0:
                break
            assert len(new_remaining_positives) < len(remaining_positives)
            remaining_positives = new_remaining_positives
            # Update all_preconditions_to_datastores.
            assert new_preconditions not in all_preconditions_to_datastores
            all_preconditions_to_datastores[new_preconditions] = datastore
        if not all_preconditions_to_datastores:
            # If we couldn't find any preconditions, default to empty.
            assert len(remaining_positives) == len(positive_data)
            all_preconditions_to_datastores[frozenset()] = positive_data
        return all_preconditions_to_datastores

    def _run_inner_search(self, pnad: PNAD, positive_data: Datastore,
                          negative_data: Datastore) -> FrozenSet[LiftedAtom]:
        """Run inner-level search to find a single precondition set."""
        initial_state = self._get_initial_preconditions(positive_data)
        check_goal = lambda s: False
        heuristic = functools.partial(self._score_preconditions, pnad,
                                      positive_data, negative_data)
        max_expansions = CFG.cluster_and_search_inner_search_max_expansions
        timeout = CFG.cluster_and_search_inner_search_timeout
        path, _ = utils.run_gbfs(initial_state,
                                 check_goal,
                                 self._get_precondition_successors,
                                 heuristic,
                                 max_expansions=max_expansions,
                                 timeout=timeout)
        return path[-1]

    @staticmethod
    def _get_initial_preconditions(
            positive_data: Datastore) -> FrozenSet[LiftedAtom]:
        """The initial preconditions are a UNION over all lifted initial states
        in the data.

        We filter out atoms containing any object that doesn't have a
        binding to the PNAD parameters.
        """
        initial_preconditions = set()
        for seg, var_to_obj in positive_data:
            obj_to_var = {v: k for k, v in var_to_obj.items()}
            for atom in seg.init_atoms:
                if not all(obj in obj_to_var for obj in atom.objects):
                    continue
                initial_preconditions.add(atom.lift(obj_to_var))
        return frozenset(initial_preconditions)

    @staticmethod
    def _get_precondition_successors(
        preconditions: FrozenSet[LiftedAtom]
    ) -> Iterator[Tuple[int, FrozenSet[LiftedAtom], float]]:
        """The successors remove each atom in the preconditions."""
        preconditions_sorted = sorted(preconditions)
        for i in range(len(preconditions_sorted)):
            successor = preconditions_sorted[:i] + preconditions_sorted[i + 1:]
            yield i, frozenset(successor), 1.0

    @staticmethod
    def _score_preconditions(pnad: PNAD, positive_data: Datastore,
                             negative_data: Datastore,
                             preconditions: FrozenSet[LiftedAtom]) -> float:
        candidate_op = pnad.op.copy_with(preconditions=preconditions)
        option_spec = pnad.option_spec
        del pnad  # unused after this
        # Count up the number of true positives and false positives.
        num_true_positives = 0
        num_false_positives = 0
        for seg, var_to_obj in positive_data:
            ground_pre = {a.ground(var_to_obj) for a in preconditions}
            if ground_pre.issubset(seg.init_atoms):
                num_true_positives += 1
        if num_true_positives == 0:
            # As a special case, if the number of true positives is 0, we
            # never want to accept these preconditions, so we can give up.
            return float("inf")
        for seg, _ in negative_data:
            # We don't want to use the substitution in the datastore for
            # negative_data, because in general the variables could be totally
            # different. So we consider all possible groundings that are
            # consistent with the option_spec. If, for any such grounding, the
            # preconditions hold in the segment's init_atoms, then this is a
            # false positive.
            objects = list(seg.states[0])
            option = seg.get_option()
            assert option.parent == option_spec[0]
            option_objs = option.objects
            isub = dict(zip(option_spec[1], option_objs))
            for idx, ground_op in enumerate(
                    utils.all_ground_operators_given_partial(
                        candidate_op, objects, isub)):
                # If the maximum number of groundings is reached, treat this
                # as a false positive. Doesn't really matter in practice
                # because the GBFS is going to time out anyway -- we just
                # want the code to not hang in this score function.
                if idx >= CFG.cluster_and_search_score_func_max_groundings or \
                   ground_op.preconditions.issubset(seg.init_atoms):
                    num_false_positives += 1
                    break
        tp_w = CFG.clustering_learner_true_pos_weight
        fp_w = CFG.clustering_learner_false_pos_weight
        score = fp_w * num_false_positives + tp_w * (-num_true_positives)
        # Penalize the number of variables in the preconditions.
        all_vars = {v for atom in preconditions for v in atom.variables}
        score += CFG.cluster_and_search_var_count_weight * len(all_vars)
        # Penalize the number of preconditions.
        score += CFG.cluster_and_search_precon_size_weight * len(preconditions)
        return score

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_search"


class ClusterAndIntersectSidelineSTRIPSLearner(ClusterAndIntersectSTRIPSLearner
                                               ):
    """Base class for a clustering-based STRIPS learner that does sidelining
    via hill climbing, after operator learning."""

    def _postprocessing_learn_ignore_effects(self,
                                             pnads: List[PNAD]) -> List[PNAD]:
        # Run hill climbing search, starting from original PNADs.
        path, _, _ = utils.run_hill_climbing(
            tuple(pnads), self._check_goal, self._get_sidelining_successors,
            functools.partial(self._evaluate, pnads))
        # The last state in the search holds the final PNADs.
        pnads = list(path[-1])
        # Because the PNADs have been modified, recompute the datastores.
        self._recompute_datastores_from_segments(pnads)
        # Filter out PNADs that have an empty datastore.
        pnads = [pnad for pnad in pnads if pnad.datastore]
        return pnads

    @abc.abstractmethod
    def _evaluate(self, initial_pnads: List[PNAD], s: Tuple[PNAD,
                                                            ...]) -> float:
        """Abstract evaluation/score function for search.

        Lower is better.
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def _check_goal(s: Tuple[PNAD, ...]) -> bool:
        del s  # unused
        # There are no goal states for this search; run until exhausted.
        return False

    @staticmethod
    def _get_sidelining_successors(
        s: Tuple[PNAD,
                 ...], ) -> Iterator[Tuple[None, Tuple[PNAD, ...], float]]:
        # For each PNAD/operator...
        for i in range(len(s)):
            pnad = s[i]
            _, option_vars = pnad.option_spec
            # ...consider changing each of its add effects to an ignore effect.
            for effect in pnad.op.add_effects:
                if len(pnad.op.add_effects) > 1:
                    # We don't want sidelining to result in a noop.
                    new_pnad = PNAD(
                        pnad.op.effect_to_ignore_effect(
                            effect, option_vars, "add"), pnad.datastore,
                        pnad.option_spec)
                    sprime = list(s)
                    sprime[i] = new_pnad
                    yield (None, tuple(sprime), 1.0)

            # ...consider removing it.
            sprime = list(s)
            del sprime[i]
            yield (None, tuple(sprime), 1.0)


class ClusterAndIntersectSidelinePredictionErrorSTRIPSLearner(
        ClusterAndIntersectSidelineSTRIPSLearner):
    """A STRIPS learner that uses hill climbing with a prediction error score
    function for ignore effect learning."""

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect_sideline_prederror"

    def _evaluate(self, initial_pnads: List[PNAD], s: Tuple[PNAD,
                                                            ...]) -> float:
        segments = [seg for traj in self._segmented_trajs for seg in traj]
        strips_ops = [pnad.op for pnad in s]
        option_specs = [pnad.option_spec for pnad in s]
        max_groundings = CFG.cluster_and_intersect_prederror_max_groundings
        num_true_positives, num_false_positives, _, _ = \
            utils.count_positives_for_ops(strips_ops, option_specs, segments,
                                          max_groundings=max_groundings)
        # Note: lower is better! We want more true positives and fewer
        # false positives.
        tp_w = CFG.clustering_learner_true_pos_weight
        fp_w = CFG.clustering_learner_false_pos_weight
        return fp_w * num_false_positives + tp_w * (-num_true_positives)


class ClusterAndIntersectSidelineHarmlessnessSTRIPSLearner(
        ClusterAndIntersectSidelineSTRIPSLearner):
    """A STRIPS learner that uses hill climbing with a harmlessness score
    function for ignore effect learning."""

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_intersect_sideline_harmlessness"

    def _evaluate(self, initial_pnads: List[PNAD], s: Tuple[PNAD,
                                                            ...]) -> float:
        preserves_harmlessness = self._check_harmlessness(list(s))
        if preserves_harmlessness:
            # If harmlessness is preserved, the score is the number of
            # operators that we have, minus the number of ignore effects.
            # This means we prefer fewer operators and more ignore effects.
            score = 2 * len(s)
            for pnad in s:
                score -= len(pnad.op.ignore_effects)
        else:
            # If harmlessness is not preserved, the score is an arbitrary
            # constant bigger than the total number of operators at the
            # start of the search. This is guaranteed to be worse (higher)
            # than any score that occurs if harmlessness is preserved.
            score = 10 * len(initial_pnads)
        return score
