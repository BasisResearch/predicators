"""Agent sim-learning + predicate-invention approach.

Extends ``AgentSimLearningApproach`` so the synthesizing Claude agent
also invents the symbolic predicates used for plan subgoals. The env's
predicates are stripped to a primitive allowlist (default
``{"Holding"}``), and the agent defines ``LEARNED_PREDICATES`` in a
sandboxed ``predicates.py``. Invented predicates flow through
``_get_all_predicates`` so they reach backtracking refinement, the
option model's abstraction function, and every other caller asking the
approach for its current predicates.

Predicates persist across online learning cycles: ``predicates.py`` is
preserved at the sandbox root, and every version evaluated during
synthesis (plus a final snapshot of post-eval edits) is saved to
``predicates_versions/`` as ``cycle_XXX_vers_YYY_predicates.py``.

Partial observability is not a separate approach: like every
sim-learning arm, the synthesis prompt follows
``CFG.partially_observable`` (see ``AgentSimLearningApproach``) - under
the flag the agent is taught the recurrent 5-arg rule signature
``rule(observation, latent, history, updates, params)`` and
``LATENT_INIT``, and this module appends the predicate-side latent
guidance (classifiers may take an optional ``latent`` kwarg,
auto-routed by ``Predicate.holds``). The latent *mechanics* (recurrent
MCMC fitting, the latent-threaded combined simulator riding
``State.latent`` so backtracking restores it per search node,
``LATENT_INIT`` loading and initial-latent seeding) live in
``AgentSimLearningApproach`` and activate automatically whenever the
loaded rules use the 5-arg signature, independent of the flag.

Example command (partially observable)::

    python predicators/main.py --env pybullet_boil \
        --approach agent_sim_predicate_invention --seed 0 \
        --num_train_tasks 10 --num_test_tasks 5 \
        --partially_observable True \
        --num_online_learning_cycles 2 --explorer agent_model_free
"""

import logging
import os
from typing import Any, Dict, FrozenSet, List, Set, Tuple

from predicators.agent_sdk import learn_prompts
from predicators.agent_sdk.tools import _SnapshotTarget, \
    finalize_versioned_snapshot, make_predicate_quality_loader
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.settings import CFG
from predicators.structs import Action, Predicate, State

logger = logging.getLogger(__name__)


class AgentSimPredicateInventionApproach(AgentSimLearningApproach):
    """Bilevel planning with learned simulator AND invented predicates.

    See module docstring.
    """

    # Always an allowlist here (the parent treats None as keep-all):
    # invention strips the env vocabulary down to Holding so everything
    # else must be invented. The stripping machinery
    # (_resolve_kept_names / _compute_kept_initial_predicates) lives on
    # AgentSimLearningApproach.
    KEPT_INITIAL_PREDICATE_NAMES: FrozenSet[str] = frozenset({"Holding"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._learned_predicates: Set[Predicate] = set()
        # Env goal atoms are hidden from the agent; goals are presented only
        # as natural language, so every train task must supply a goal_nl.
        missing = [i for i, t in enumerate(self._train_tasks) if not t.goal_nl]
        assert not missing, (
            f"{type(self).__name__} requires every train task to set "
            f"`goal_nl` (env goal atoms are deliberately not exposed to "
            f"the agent). Missing on task indices: {missing}")

    @classmethod
    def get_name(cls) -> str:
        return "agent_sim_predicate_invention"

    # ── Checkpointing ───────────────────────────────────────────

    # Own suffix: this subclass's checkpoints additionally rely on the
    # restored predicates.py (invented predicates rehydrate from it).
    _save_suffix: str = "AgentSimPredInv"

    def _rehydrate_extra_artifacts(self, base: str) -> None:
        """Reload invented predicates from the restored predicates.py.

        Called by the parent AFTER ``_fitted_params`` is restored, so
        predicate lambdas closing over ``params[...]`` see the fitted
        values.
        """
        predicates_file = os.path.join(base, "predicates.py")
        if not os.path.isfile(predicates_file):
            return
        self._learned_predicates = self._load_predicates_from_module_file(
            predicates_file)
        logger.info("Rehydrated %d learned predicate(s) from checkpoint.",
                    len(self._learned_predicates))

    # ── Predicate set ───────────────────────────────────────────

    def _get_all_predicates(self) -> Set[Predicate]:
        return super()._get_all_predicates() | self._learned_predicates

    # ── Agent session hooks ─────────────────────────────────────

    # ── Synthesis hooks ──────────────────────────────────────────

    def _compute_extra_synthesis_paths(self, base: str) -> Dict[str, str]:
        predicates_file = os.path.join(base, "predicates.py")
        predicates_versions_dir = os.path.join(base, "predicates_versions")

        if CFG.agent_sdk_use_local_sandbox:
            predicates_file_for_agent = "./predicates.py"
        elif self._tool_context.sandbox_dir:
            predicates_file_for_agent = "/sandbox/predicates.py"
        else:
            predicates_file_for_agent = predicates_file

        return {
            "predicates_file": predicates_file,
            "predicates_versions_dir": predicates_versions_dir,
            "predicates_file_for_agent": predicates_file_for_agent,
        }

    def _install_extra_synthesis_surfaces(
        self,
        exec_ns: Dict[str, Any],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
        extra_paths: Dict[str, str],
    ) -> None:
        del exec_ns, base_pred_triples, inferred_hint
        self._tool_context.probe_artifact_loaders["predicates"] = \
            make_predicate_quality_loader(
                predicates_file=extra_paths["predicates_file"],
                predicates_versions_dir=extra_paths[
                    "predicates_versions_dir"],
                approach=self,
                trajectories=self._get_all_trajectories(),
                cycle_index_provider=self._learning_cycle_index,
            )

    def _build_write_snapshot_targets(
        self,
        simulator_file: str,
        versions_dir: str,
        extra_paths: Dict[str, str],
    ) -> List[_SnapshotTarget]:
        targets = super()._build_write_snapshot_targets(
            simulator_file, versions_dir, extra_paths)
        targets.append(
            _SnapshotTarget(
                live_file=extra_paths["predicates_file"],
                versions_dir=extra_paths["predicates_versions_dir"],
                artifact_name="predicates",
                cycle_index_provider=self._learning_cycle_index,
            ))
        return targets

    def _extra_synthesis_message(self, extra_paths: Dict[str, str]) -> str:
        message = learn_prompts.render_predicate_invention_message(
            extra_paths["predicates_file_for_agent"],
            self._format_goal_nl_block())
        return message + self._chained_extra_message(extra_paths)

    def _chained_extra_message(self, extra_paths: Dict[str, str]) -> str:
        """The base class's extra message (the partial-observability note under
        ``CFG.partially_observable``), separated for appending."""
        base = super()._extra_synthesis_message(extra_paths)
        return "\n\n" + base if base else ""

    def _format_goal_nl_block(self) -> str:
        """Render the deduped natural-language goals for the train tasks.

        Returns an empty string only if every task is missing a
        ``goal_nl``, but ``__init__`` asserts they're present, so in
        practice this always returns a non-empty block.
        """
        seen: List[str] = []
        for task in self._train_tasks:
            nl = task.goal_nl
            if nl and nl not in seen:
                seen.append(nl)
        if not seen:
            return ""
        if len(seen) == 1:
            return f"Goal (natural language): {seen[0]}\n\n"
        bullets = "\n".join(f"  - {g}" for g in seen)
        return f"Goals across train tasks (natural language):\n{bullets}\n\n"

    def _synthesis_workflow_extra(self) -> str:
        # The base workflow's validation step depends on invented
        # predicates: sketches can only reference predicates that exist.
        return learn_prompts.render_predicate_workflow_extra()

    def _extra_synthesis_system_prompt_sections(self) -> List[str]:
        # The scene workbench is the sim probe inside run_python (the
        # probe is unconditional in synthesis sessions).
        workbench = ("the `sim` probe in `run_python` as scene workbench "
                     "(`sim.reset(task_idx=..., mods={...})` to stage "
                     "states, `sim.render(label, annotations=[...])` "
                     "to render with overlays)")
        sections = super()._extra_synthesis_system_prompt_sections()
        sections.append(
            learn_prompts.render_predicate_invention_section(workbench))
        return sections

    def _extra_synthesis_latent_sections(self) -> List[str]:
        # The predicate-side latent guidance belongs to invention arms
        # only and follows the simulator-side tutorial it refers to.
        sections = super()._extra_synthesis_latent_sections()
        sections.append(learn_prompts.render_predicate_latent_section())
        return sections

    def _post_synthesis_loading(
        self,
        extra_paths: Dict[str, str],
        specs: List[Any],
    ) -> None:
        """Load predicates.py and snapshot the cycle's final state."""
        predicates_file = extra_paths["predicates_file"]
        predicates_versions_dir = extra_paths["predicates_versions_dir"]

        # Seed _fitted_params from init values so predicate lambdas
        # closing over ``params["..."]`` are evaluable during validation.
        # The real MCMC fit runs later in the base flow and overwrites
        # these. Mutate in place so _ParamsView holders pick up the seeds.
        if specs:
            self._fitted_params.clear()
            self._fitted_params.update({s.name: s.init_value for s in specs})

        final_pred_tag = finalize_versioned_snapshot(
            predicates_file,
            predicates_versions_dir,
            cycle_idx=self._learning_cycle_index(),
            artifact_name="predicates",
        )
        if final_pred_tag is not None:
            self._current_predicates_version = final_pred_tag
            logger.info("Final predicates snapshot: %s", final_pred_tag)

        loaded = self._load_predicates_from_module_file(predicates_file)
        self._learned_predicates = loaded
        logger.info("Loaded %d learned predicate(s) from %s.", len(loaded),
                    predicates_file)
        for p in sorted(loaded, key=lambda x: x.name):
            sig = ", ".join(t.name for t in p.types)
            logger.info("  %s(%s)", p.name, sig)

    # ── Predicate loading ────────────────────────────────────────

    def _load_predicates_from_module_file(self, path: str) -> Set[Predicate]:
        """Load LEARNED_PREDICATES from ``path``; validate each.

        Mirrors the simulator-loader pattern. Returns the empty set on
        missing file or exec failure (predicates are optional). Skips
        and warns on entries that fail validation or collide with kept
        env predicate names.
        """
        # pylint: disable=import-outside-toplevel
        from predicators.agent_sdk.proposal_exec import build_exec_context, \
            exec_code_safely, validate_predicate
        from predicators.agent_sdk.tools import _ParamsView
        from predicators.code_sim_learning.fit_space import ParamSpec

        # pylint: enable=import-outside-toplevel

        if not os.path.isfile(path):
            logger.info("No predicates file at %s; learned set is empty.",
                        path)
            return set()

        with open(path, "r", encoding="utf-8") as f:
            code = f.read()

        ctx = build_exec_context(types=self._types,
                                 predicates=self._kept_initial_predicates,
                                 options=self._get_all_options(),
                                 extra_context={
                                     "params":
                                     _ParamsView(self._fitted_params),
                                     "ParamSpec": ParamSpec,
                                 })

        result, err = exec_code_safely(code, ctx, "LEARNED_PREDICATES")
        if err is not None:
            logger.warning("Failed to load %s:\n%s", path, err)
            return set()
        if not isinstance(result, list):
            logger.warning("%s: LEARNED_PREDICATES must be a list, got %s.",
                           path,
                           type(result).__name__)
            return set()

        kept_names = {p.name for p in self._kept_initial_predicates}
        example_state = (self._train_tasks[0].init
                         if self._train_tasks else None)

        valid: Set[Predicate] = set()
        seen_names: Set[str] = set()
        for entry in result:
            if not isinstance(entry, Predicate):
                logger.warning("Skipped non-Predicate entry in %s: %r", path,
                               entry)
                continue
            if entry.name in kept_names:
                logger.warning(
                    "Skipped '%s' (collides with a kept env predicate).",
                    entry.name)
                continue
            if entry.name in seen_names:
                logger.warning("Skipped duplicate '%s' in %s.", entry.name,
                               path)
                continue
            if example_state is not None:
                verr = validate_predicate(entry, self._types, example_state)
                if verr is not None:
                    logger.warning("Predicate '%s' validation failed: %s",
                                   entry.name, verr)
                    continue
            valid.add(entry)
            seen_names.add(entry.name)

        return valid
