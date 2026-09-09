"""The natural-language world model baseline (paper arm C3).

The same loop and experiments as the code arms - the agent explores the
train tasks, learns after every cycle, and solves the test tasks - but
the learned model is a natural-language document, ``world_model.md``,
never executable code. The learn session writes it from the recorded
data with the same data tools the code arms get; the solve and explore
sessions receive it quoted into every task message and plan by
reasoning over it, with no simulator to test plans against (the
model-free planner's open-loop solve). Predicates are the env's kept
initial ones (the same allowlist as the code arms) and goals arrive as
natural language.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from predicators.agent_sdk import learn_prompts
from predicators.agent_sdk.session_base import AgentSessionFatalError, \
    query_fatal_error
from predicators.agent_sdk.tools import _SnapshotTarget
from predicators.agent_sdk.tools.digests import render_options_digest, \
    render_trajectory_digest, render_types_digest
from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result
from predicators.agent_sdk.tools.snapshots import finalize_versioned_snapshot
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.code_sim_learning.program_world_model import \
    option_transitions
from predicators.settings import CFG
from predicators.structs import Dataset, InteractionResult, \
    LowLevelTrajectory, Predicate

logger = logging.getLogger(__name__)

_NOTES_FILE = "world_model.md"
_NOTES_VERSIONS_DIR = "world_model_versions"

_RUN_PYTHON_DESCRIPTION = (
    "Execute Python code (`code`, or `path` to a .py file you wrote in "
    "the sandbox) for data exploration. Available variables: "
    "trajectories (List[LowLevelTrajectory]; each has `is_demo`, "
    "`train_task_idx`, `states`, `actions`; each action's `get_option()` "
    "is the skill that produced it), train_tasks (List[Task]; each has "
    "`init`, `goal`, `goal_holds(state)`), is_goal_state (callable: "
    "state, task_idx -> bool), describe_trajectory(traj_idx, "
    "include_states=True, include_atoms=False, max_timesteps=10), and "
    "np. print() output is returned; the namespace persists across "
    "calls; oversize output is saved under `tool_outputs/run_python/` "
    "and previewed. There is no simulator in this session: the world "
    "model you write is a document, checked by predicting recorded "
    "transitions from it by hand.")


class AgentNotesWorldModelApproach(AgentModelFreeApproach):
    """Model-free agentic planning over a learned natural-language world model
    document."""

    _save_suffix: str = "AgentNotesWM"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._notes: str = ""
        self._notes_version: Optional[str] = None
        missing = [i for i, t in enumerate(self._train_tasks) if not t.goal_nl]
        assert not missing, (
            f"{type(self).__name__} presents goals in natural language, so "
            f"every train task must set `goal_nl`. Missing on task "
            f"indices: {missing}")

    @classmethod
    def get_name(cls) -> str:
        return "agent_nl_world_model"

    # ── Vocabulary ───────────────────────────────────────────────

    def _get_all_predicates(self) -> Set[Predicate]:
        """The env predicates, restricted to the configured allowlist
        (``agent_sim_learn_kept_predicates_names``) like the code arms."""
        preds = super()._get_all_predicates()
        kept = CFG.agent_sim_learn_kept_predicates_names
        if kept:
            preds = {p for p in preds if p.name in set(kept)}
        return preds

    # ── Session surface ──────────────────────────────────────────

    def _get_synthesis_tool_names(self) -> Optional[List[str]]:
        return ["run_python"]

    def _get_agent_system_prompt(self) -> str:
        if self._learning_mode:
            return learn_prompts.build_notes_learn_system_prompt()
        return "\n\n".join([
            super()._get_agent_system_prompt(),
            learn_prompts.render_notes_solve_system_section(),
        ])

    def _solve_prompt_extra_sections(self) -> str:
        return learn_prompts.render_world_model_notes_block(
            self._notes,
            self._notes_paths()["notes_file_for_agent"])

    def _sync_tool_context(self) -> None:
        super()._sync_tool_context()
        self._tool_context.world_model_notes = self._notes
        self._tool_context.world_model_notes_path = \
            self._notes_paths()["notes_file_for_agent"]

    # ── Learning ─────────────────────────────────────────────────

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        super().learn_from_offline_dataset(dataset)
        self._learn_notes()
        self.save(None)

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        cycle = self._online_learning_cycle
        super().learn_from_interaction_results(results)
        self._learn_notes()
        self.save(cycle)

    def _checkpoint_after_offline_learning(self) -> None:
        """No-op: this class checkpoints after its own learning."""

    def _checkpoint_after_interaction_results(self, cycle: int) -> None:
        """No-op: this class checkpoints after its own learning."""
        del cycle

    def _learn_notes(self) -> None:
        """Run one document-writing session over all recorded data."""
        trajectories = self._get_all_trajectories()
        if not trajectories and not CFG.agent_sim_learn_zero_shot:
            logger.warning("No recorded trajectories; skipping the world "
                           "model document session.")
            return
        if not trajectories:
            logger.info("Zero-shot synthesis: the agent writes the world "
                        "model document without data.")
        self._run_notes_session(trajectories)

    def _notes_paths(self) -> Dict[str, str]:
        """Host and agent-visible paths of the document (the residual arm's
        sandbox mapping)."""
        if CFG.agent_sdk_use_local_sandbox:
            sandbox_dir: Optional[str] = os.path.abspath(
                os.path.join(self._get_log_dir(), "sandbox"))
        else:
            sandbox_dir = self._tool_context.sandbox_dir
        base = sandbox_dir or self._get_log_dir()
        notes_file = os.path.join(base, _NOTES_FILE)
        if CFG.agent_sdk_use_local_sandbox:
            notes_file_for_agent = f"./{_NOTES_FILE}"
            sandbox_dir_for_agent: Optional[str] = "."
        elif sandbox_dir:
            notes_file_for_agent = f"/sandbox/{_NOTES_FILE}"
            sandbox_dir_for_agent = "/sandbox"
        else:
            notes_file_for_agent = notes_file
            sandbox_dir_for_agent = None
        return {
            "base": base,
            "notes_file": notes_file,
            "versions_dir": os.path.join(base, _NOTES_VERSIONS_DIR),
            "notes_file_for_agent": notes_file_for_agent,
            "sandbox_dir_for_agent": sandbox_dir_for_agent or "",
        }

    def _build_notes_exec_ns(
            self, trajectories: List[LowLevelTrajectory]) -> Dict[str, Any]:
        predicates = self._get_all_predicates()
        train_tasks = self._train_tasks

        def describe_trajectory(traj_idx: int,
                                include_states: bool = True,
                                include_atoms: bool = False,
                                max_timesteps: int = 10) -> str:
            return render_trajectory_digest(trajectories,
                                            train_tasks,
                                            predicates,
                                            traj_idx,
                                            include_states=include_states,
                                            include_atoms=include_atoms,
                                            max_timesteps=max_timesteps)

        return {
            "trajectories":
            trajectories,
            "train_tasks":
            train_tasks,
            "is_goal_state":
            lambda state, task_idx: train_tasks[task_idx].goal_holds(state),
            "describe_trajectory":
            describe_trajectory,
            "np":
            np,
        }

    def _run_notes_session(self,
                           trajectories: List[LowLevelTrajectory]) -> None:
        # pylint: disable=import-outside-toplevel
        from claude_agent_sdk import tool as _sdk_tool

        from predicators.approaches.agent_sim_learning_approach import \
            AgentSimLearningApproach

        # pylint: enable=import-outside-toplevel
        paths = self._notes_paths()
        os.makedirs(paths["base"], exist_ok=True)
        # A restored document is written back so the agent can Read it.
        if self._notes and not os.path.isfile(paths["notes_file"]):
            with open(paths["notes_file"], "w", encoding="utf-8") as f:
                f.write(self._notes)
        exec_ns = self._build_notes_exec_ns(trajectories)
        run_python = _make_python_exec_tool(
            _make_coercing_tool(_sdk_tool),
            name="run_python",
            description=_RUN_PYTHON_DESCRIPTION,
            exec_ns=exec_ns,
            sandbox_dir=paths["base"],
            sandbox_dir_for_agent=paths["sandbox_dir_for_agent"] or None,
            text_result=_make_spilling_text_result(
                paths["base"],
                agent_prefix=paths["sandbox_dir_for_agent"] or None),
            call_timeout_s=CFG.agent_sdk_synthesis_python_call_timeout,
        )
        ctx = self._tool_context
        ctx.extra_mcp_tools = [run_python]
        ctx.learn_cycle_index = self._online_learning_cycle
        targets = [
            _SnapshotTarget(
                live_file=paths["notes_file"],
                versions_dir=paths["versions_dir"],
                artifact_name="world_model_notes",
                cycle_index_provider=lambda: self._online_learning_cycle,
            )
        ]
        build_hooks = (
            AgentSimLearningApproach._build_synthesis_session_hooks  # pylint: disable=protected-access
        )
        ctx.extra_session_hooks = build_hooks(targets, paths["base"])
        self._learning_mode = True
        self._close_agent_session()
        try:
            self._ensure_agent_session()
            message = self._build_notes_learn_message(trajectories, paths)
            responses = self._query_agent_sync(message, kind="learn")
            dead = query_fatal_error(responses)
            if dead is not None:
                raise AgentSessionFatalError(
                    "The learn session died without the agent doing any "
                    f"work ({dead}); refusing to checkpoint this cycle as "
                    "learned.")
        finally:
            ctx.extra_session_hooks = {}
            ctx.extra_mcp_tools = []
            ctx.learn_cycle_index = None
            self._learning_mode = False
            self._close_agent_session()
        self._load_notes(paths)

    def _build_notes_learn_message(self,
                                   trajectories: List[LowLevelTrajectory],
                                   paths: Dict[str, str]) -> str:
        predicates = self._get_all_predicates()
        n_trajs = len(trajectories)
        n_demos = sum(1 for t in trajectories if t.is_demo)
        n_transitions = sum(
            len(option_transitions(t, predicates)) for t in trajectories)
        session_tool_names = (self._agent_session.tool_names
                              if self._agent_session is not None else [])
        extra_messages: List[str] = []
        if not trajectories and CFG.agent_sim_learn_zero_shot:
            extra_messages.append(
                learn_prompts.render_notes_zero_shot_message())
        listing = "\n".join(
            f"  [{i}] {'demo' if t.is_demo else 'interaction'}, task "
            f"{t.train_task_idx}" for i, t in enumerate(trajectories))
        signatures = "\n".join(
            f"- {p.name}({', '.join(t.name for t in p.types)})"
            for p in sorted(predicates, key=lambda p: p.name))
        objective = next((t.evaluator.objective_description()
                          for t in self._train_tasks if t.evaluator is not None
                          and t.evaluator.objective_description()), "")
        return learn_prompts.build_notes_learn_message(
            n_trajs=n_trajs,
            n_transitions=n_transitions,
            n_demos=n_demos,
            n_interaction=n_trajs - n_demos,
            trajectory_listing=listing,
            structs_ref=self._structs_reference_path(),
            predicate_listing=signatures or "(none)",
            types_digest=render_types_digest(self._tool_context.types),
            options_digest=render_options_digest(
                self._get_all_options(),
                gt_options_ref_path=self._tool_context.gt_options_ref_path),
            notes_file=paths["notes_file_for_agent"],
            goal_nls=[t.goal_nl or "" for t in self._train_tasks],
            has_prior_notes=os.path.isfile(paths["notes_file"]),
            objective_block=learn_prompts.render_objective_block(objective),
            tools_block=learn_prompts.render_tools_block(session_tool_names),
            extra_messages=extra_messages,
        )

    def _structs_reference_path(self) -> str:
        """Write the data-structures source into the sandbox reference dir (the
        residual arm's convention) and return its agent-visible path."""
        # pylint: disable-next=import-outside-toplevel
        import inspect

        # pylint: disable-next=import-outside-toplevel
        from predicators import structs
        paths = self._notes_paths()
        ref_dir = os.path.join(paths["base"], "reference")
        os.makedirs(ref_dir, exist_ok=True)
        with open(os.path.join(ref_dir, "structs.py"), "w",
                  encoding="utf-8") as f:
            f.write(inspect.getsource(structs))
        if paths["sandbox_dir_for_agent"]:
            return f"{paths['sandbox_dir_for_agent']}/reference/structs.py"
        return os.path.join(ref_dir, "structs.py")

    def _load_notes(self, paths: Dict[str, str]) -> None:
        tag = finalize_versioned_snapshot(
            paths["notes_file"],
            paths["versions_dir"],
            cycle_idx=self._online_learning_cycle,
            artifact_name="world_model_notes")
        if not os.path.isfile(paths["notes_file"]):
            logger.warning(
                "The session left no %s; the previous document "
                "stands.", _NOTES_FILE)
            return
        with open(paths["notes_file"], "r", encoding="utf-8") as f:
            self._notes = f.read()
        self._notes_version = tag
        self._sync_tool_context()
        logger.info("Loaded the world model document (%d chars, %s).",
                    len(self._notes), tag or "unversioned")

    # ── Checkpointing ────────────────────────────────────────────

    def _extra_save_state(self) -> Dict[str, Any]:
        return {
            "world_model_notes": self._notes,
            "world_model_notes_version": self._notes_version,
        }

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        self._notes = str(save_dict.get("world_model_notes") or "")
        self._notes_version = save_dict.get("world_model_notes_version")
        if self._notes:
            paths = self._notes_paths()
            os.makedirs(paths["base"], exist_ok=True)
            with open(paths["notes_file"], "w", encoding="utf-8") as f:
                f.write(self._notes)
