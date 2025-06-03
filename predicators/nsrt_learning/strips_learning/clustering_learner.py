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
import bisect

from predicators import utils
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.planning import PlanningFailure, PlanningTimeout, \
    task_plan_grounding
from predicators.planning_with_processes import \
    task_plan as task_plan_with_processes
from predicators.settings import CFG
from predicators.structs import PNAD, Datastore, DummyOption, \
    EndogenousProcess, ExogenousProcess, LiftedAtom, ParameterizedOption, \
    Predicate, Segment, STRIPSOperator, VarToObjSub, Variable, Object


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
                ] or CFG.exogenous_process_learner_do_intersect:
                    preconds1 = frozenset()  # no preconditions
                    preconds2 = frozenset()  # no preconditions
                else:
                    # Ground
                    preconds1 = frozenset(segment.init_atoms)
                    # Lifted
                    obj_to_var = {
                        v: k
                        for k, v in pnad.datastore[-1][1].items()
                    }
                    preconds2 = frozenset({
                        atom.lift(obj_to_var)
                        for atom in pnad.datastore[-1][0].init_atoms
                    })
                # ent_to_ent_sub here is obj_to_var
                suc, ent_to_ent_sub = utils.unify_preconds_effects_options(
                    preconds1, preconds2, frozenset(segment.add_effects),
                    frozenset(pnad.op.add_effects),
                    frozenset(segment.delete_effects),
                    frozenset(pnad.op.delete_effects), segment_param_option,
                    pnad_param_option, segment_option_objs,
                    tuple(pnad_option_vars))
                sub = cast(VarToObjSub,
                           {v: o for o, v in ent_to_ent_sub.items()})
                if suc:
                    # Add to this PNAD.
                    if CFG.exogenous_process_learner_do_intersect:
                        # Find the largest conditions that unifies the init
                        # atoms of the segment and another segment in the PNAD.
                        # and add that segment and sub to the datastore.
                        # Doing this sequentially ensures one of the 
                        # substitutions has the objects we care about with
                        # intersection. Hence it can fall out later in
                        # `induce_preconditions_via_intersection`.
                        sub = self._maybe_intersect_segment_with_pnad(
                            segment, pnad, ent_to_ent_sub,
                            segment_param_option, pnad_param_option,
                            segment_option_objs, tuple(pnad_option_vars))
                    else:
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

    def _maybe_intersect_segment_with_pnad(
        self,
        segment: Segment,
        pnad: PNAD,
        obj_to_var: Dict[Object, Variable],
        segment_param_option: ParameterizedOption,
        pnad_param_option: ParameterizedOption,
        segment_option_objs: Tuple[Object],
        pnad_option_vars: Tuple[Variable]
    ) -> VarToObjSub:
        """Try to unify and find the largest conditions that unify the init atoms
        of the segment and the last segment in the pnad datastore. Returns an
        updated VarToObjSub."""
        seg_init_atoms_full = set(segment.init_atoms)

        last_seg, last_var_to_obj = pnad.datastore[-1]
        last_obj_to_var = {o: v for v, o in last_var_to_obj.items()}
        lifted_last_init_atoms = {
            atom.lift(last_obj_to_var) for atom in last_seg.init_atoms
        }

        # Candidate atoms that possibly match
        common_preds = {a.predicate for a in seg_init_atoms_full} & \
                    {b.predicate for b in lifted_last_init_atoms}

        s_init_atoms_list = sorted(
            [atom for atom in seg_init_atoms_full if atom.predicate in common_preds],
            key=str
        )
        ds_lifted_init_atoms_list = sorted(
            [atom for atom in lifted_last_init_atoms if atom.predicate in common_preds],
            key=str
        )
        max_len1 = len(s_init_atoms_list)
        max_len2 = len(ds_lifted_init_atoms_list)

        seg_add_eff = frozenset(segment.add_effects)
        pnad_add_eff = frozenset(pnad.op.add_effects)
        seg_del_eff = frozenset(segment.delete_effects)
        pnad_del_eff = frozenset(pnad.op.delete_effects)

        unify_args = (
            seg_add_eff, pnad_add_eff,
            seg_del_eff, pnad_del_eff,
            segment_param_option, pnad_param_option,
            segment_option_objs, tuple(pnad_option_vars)
        )

        best_obj_to_var = obj_to_var
        found_best_unification = False
        k_limit = min(max_len1, max_len2)

        for k_common in range(k_limit, -1, -1):
            for p1_subset_tuple in itertools.combinations(s_init_atoms_list, k_common):
                p1_candidate = frozenset(p1_subset_tuple)
                for p2_subset_tuple in itertools.combinations(ds_lifted_init_atoms_list, k_common):
                    p2_candidate = frozenset(p2_subset_tuple)

                    # Check if they have the same predicates
                    if {a.predicate for a in p1_candidate} != {a.predicate for a in p2_candidate}:
                        continue

                    # Check if they unify
                    current_suc, current_obj_to_var = utils.unify_preconds_effects_options(
                        p1_candidate, p2_candidate, *unify_args
                    )
                    if current_suc:
                        best_obj_to_var = current_obj_to_var
                        found_best_unification = True
                        break
                if found_best_unification:
                    break
            if found_best_unification:
                break

        sub = cast(VarToObjSub, {v: o for o, v in best_obj_to_var.items()})
        return sub

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
            get_false_positive_states
        self._get_false_positive_process_states = get_false_positive_states

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
        from predicators.approaches.pp_online_predicate_invention_approach \
            import get_false_positive_states_from_seg_trajs
        self._get_false_positive_states_from_seg_trajs = \
            get_false_positive_states_from_seg_trajs
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
        return []

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

    def _run_search(self, pnad: PNAD) -> FrozenSet[LiftedAtom]:

        logging.info(f"For operator:\n{pnad.op}")
        init_ground_atoms = pnad.datastore[0][0].init_atoms
        var_to_obj = pnad.datastore[0][1]
        obj_to_var = {v: k for k, v in var_to_obj.items()}
        initial_state: FrozenSet[LiftedAtom] = frozenset(
            atom.lift(obj_to_var) for atom in init_ground_atoms)
        delay_param_estim = [len(pnad.datastore[0][0].actions)]
        exogenous_process = pnad.make_exogenous_process(
            process_delay_params=delay_param_estim
        )
        score_func = functools.partial(self._score_preconditions,
                                       exogenous_process)

        path, _ = utils.run_gbfs(initial_state, lambda s: False,
                                 self._get_preconditions_successors,
                                 score_func)

        return_precon = path[-1]
        logging.debug(f"Search finished. Selected:")
        score_func(return_precon)
        return return_precon

    def _score_preconditions(self, exogenous_process: ExogenousProcess,
                             preconditions: FrozenSet[LiftedAtom]) -> float:
        exogenous_process.condition_at_start = set(preconditions)
        exogenous_process.condition_overall = set(preconditions)
        false_positive_process_state =\
            self._get_false_positive_states_from_seg_trajs(
                self._atom_change_segmented_trajs, [exogenous_process])
        num_false_positives = 0
        for _, states in false_positive_process_state.items():
            num_false_positives += len(states)
            # logging.debug(states)

        complexity_penalty = CFG.grammar_search_pred_complexity_weight *\
                                    len(preconditions)
        cost = num_false_positives + complexity_penalty
        logging.debug(f"Condition: {set(preconditions)}, Score {cost:.4f}")
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._endogenous_processes = kwargs["endogenous_processes"]

        from predicators.predicate_search_score_functions import \
            _ExpectedNodesScoreFunction
        self._get_optimality_prob =\
            _ExpectedNodesScoreFunction._get_refinement_prob

        from predicators.approaches.pp_online_predicate_invention_approach import \
            get_false_positive_states_from_seg_trajs
        self._get_fp_states_from_seg_trajs = \
            get_false_positive_states_from_seg_trajs

        self._atom_change_segmented_trajs: List[List[Segment]] = []
        self._option_change_segmented_trajs: List[List[Segment]] = []
        self._demo_atoms_sequences: List[List[Set[LiftedAtom]]] = []
        self._total_num_candidates = 0

    @classmethod
    def get_name(cls) -> str:
        return "cluster_and_inverse_planning"

    def _learn_pnad_preconditions(self, pnads: List[PNAD]) -> List[PNAD]:
        """Find the set of PNADs (with corresponding processes) that allows the
        agent make similar plans as the demonstrated/successful plans."""

        self._total_num_candidates = 0
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
        self._demo_atoms_sequences = [
                utils.segment_trajectory_to_atoms_sequence(seg_traj)
            for seg_traj in self._option_change_segmented_trajs]

        # --- Get the candidate preconditions ---
        # First option. Candidates are all possible subsets.
        conditions_at_start = []
        for pnad in pnads:
            if CFG.exogenous_process_learner_do_intersect:
                init_lift_atoms = self._induce_preconditions_via_intersection(
                    pnad)
            else:
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
                conditions_at_start.append(
                    self._get_top_consistent_conditions(init_lift_atoms, pnad))
            else:
                raise NotImplementedError

        # --- Search for the best combination of preconditions ---
        best_cost = float("inf")
        best_conditions = []
        # Score all combinations of preconditions
        for i, combination in enumerate(itertools.product(*conditions_at_start)):
            # Set the conditions for each process
            for process, conditions in zip(exogenous_process, combination):
                process.condition_at_start = conditions
                process.condition_overall = conditions

            # Score this set of processes
            cost = self.compute_processes_score(set(exogenous_process))
            if cost < best_cost:
                best_cost = cost
                best_conditions = combination
            logging.debug(
                f"Combination {i+1}/{self._total_num_candidates}: cost = {cost},"
                f" Best cost = {best_cost}")

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
                new_pnad = PNAD(pnad.op.copy_with(preconditions=conditions),
                                pnad.datastore, pnad.option_spec)
                final_pnads.append(new_pnad)
        return final_pnads

    def _get_top_consistent_conditions(self, initial_atom: Set[LiftedAtom],
                                      pnad: PNAD) -> Iterator[Set[LiftedAtom]]:
        """Get the top consistent conditions for a PNAD."""
        # TODO: maybe a better way is to based on percentage of the worse score
        # because as the number of trajectories increases, the worse score
        # increases
        exogenous_process = pnad.make_exogenous_process()

        method = CFG.cluster_and_inverse_planning_top_consistent_method

        if method == "threshold":
            # Original threshold-based approach
            for condition_candidate in utils.all_subsets(initial_atom):
                exogenous_process.condition_at_start = condition_candidate
                exogenous_process.condition_overall = condition_candidate

                false_positive_process_state = \
                    self._get_fp_states_from_seg_trajs(
                        self._atom_change_segmented_trajs, [exogenous_process])
                num_false_positives = sum(
                    len(states)
                    for states in false_positive_process_state.values())

                logging.debug(
                    f"Conditions: {condition_candidate}, FP: {num_false_positives}"
                )
                if num_false_positives <= CFG.cluster_and_inverse_planning_top_consistent_max_cost:
                    yield condition_candidate
        # return candidates
        elif method in ["top_p_percent", "top_n"]:
            # Collect all candidates with their scores
            candidates_with_scores = []

            logging.info(f"For operator sketch:\n{pnad.op}")
            for condition_candidate in utils.all_subsets(initial_atom):
                exogenous_process.condition_at_start = condition_candidate
                exogenous_process.condition_overall = condition_candidate

                false_positive_process_state = \
                    self._get_fp_states_from_seg_trajs(
                        self._atom_change_segmented_trajs, [exogenous_process])
                num_false_positives = sum(
                    len(states)
                    for states in false_positive_process_state.values())

                # Add complexity penalty for tie-breaking (prefer simpler conditions)
                complexity_penalty =\
                    CFG.grammar_search_pred_complexity_weight * len(
                    condition_candidate)
                score = num_false_positives + complexity_penalty

                candidates_with_scores.append((score, condition_candidate))
                logging.debug(
                    f"Conditions: {condition_candidate}, Score: {score}")

            # Sort by score (lower is better)
            candidates_with_scores.sort(key=lambda x: x[0])

            if method == "top_p_percent":
                # Return top p% of candidates
                p_percent = CFG.cluster_and_inverse_planning_top_p_percent
                n_candidates = len(candidates_with_scores)
                num_under_percentage = max(
                    1, int(n_candidates * p_percent / 100.0))
                score_at_threshold = candidates_with_scores[:num_under_percentage][-1][0]
                scores = [score for score, _ in candidates_with_scores]
                # Include all candidates with score_at_threshold
                position = bisect.bisect_right(scores, score_at_threshold)
                # scores = [score for score, _ in candidates_with_scores]
                # # This gets the insertion point to keep list sorted if "target" were added
                # position =  bisect.bisect_right(scores, 0.004)
                # last_leq_index = position
                # return_idx = max(last_leq_index, num_under_percentage)
                # last_top_score = candidates_with_scores[num_under_percentage-1
                #                                         ][0]
                logging.info(f"Score threshold {score_at_threshold}; returning "
                        f"{position}/{n_candidates} candidates")
                
                # Reocrd the total number of candidates
                if self._total_num_candidates == 0:
                    self._total_num_candidates += position
                else:
                    self._total_num_candidates *= position
                top_candidates = candidates_with_scores[:position]
            else:  # method == "top_n"
                # Return top n candidates
                n = CFG.cluster_and_inverse_planning_top_n
                top_candidates = candidates_with_scores[:n]

            # Yield the selected candidates
            for score, condition_candidate in top_candidates:
                logging.info(
                    f"Selected condition: {condition_candidate}, Score: {score}")
                yield condition_candidate

        else:
            raise ValueError(
                f"Unknown method: {method}. Must be one of 'threshold', "
                "'top_p_percent', 'top_n'"
            )

    def compute_processes_score(
            self, exogenous_processes: Set[ExogenousProcess]) -> float:
        """Score the PNAD based on how well it allows the agent to make
        plans."""
        # TODO: also incorporate number of nodes expanded to the function
        cost = 0.0
        for i, traj in enumerate(self._trajectories):
            if not traj.is_demo:
                continue
            demo_atoms_sequence = self._demo_atoms_sequences[i]
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
                ground_processes, self._predicates, objects)
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
            num_nodes = CFG.grammar_search_expected_nodes_upper_bound
            try:
                for idx, (_, plan_atoms_sequence,
                          metrics) in enumerate(generator):
                    num_nodes = metrics["num_nodes_created"]
                    optimality_prob = self._get_optimality_prob(
                        demo_atoms_sequence, plan_atoms_sequence)
            except (PlanningTimeout, PlanningFailure):
                pass
            # low_quality_prob = 1.0 - optimality_prob
            cost += (1 - optimality_prob) # * num_nodes

        return cost


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
