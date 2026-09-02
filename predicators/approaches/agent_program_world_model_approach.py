"""Bilevel planning with an agent-written option-level program world model and
invented predicates (paper arm C4: a code world model with no engine
underneath, in the form of Pinductor / POMDP Coder).

Everything about the loop is the residual arm's - the explorer, the
sketch / refine / run tools, the capture gate, the solve pipeline,
predicate invention - except the model artifact: instead of residual
rules over a physics engine the agent writes ``world_model.py``, an
option-level transition program with its own hidden state (see
:mod:`code_sim_learning.program_world_model`). There is no parameter
fit; the learn session scores the program with the Pinductor
particle-filter kernel pseudo-likelihood (``sim.score``) and the agent
edits it. The belief over the hidden state is a particle set drawn from
the program's ``initial_latent``; the capture gate re-rolls every
submission under every particle through the residual arm's
rule-parameter margin channel.
"""
from __future__ import annotations

import copy
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np

from predicators import utils
from predicators.agent_sdk import learn_prompts
from predicators.agent_sdk.session_base import AgentSessionFatalError, \
    query_fatal_error
from predicators.agent_sdk.tools import _SnapshotTarget
from predicators.agent_sdk.tools.digests import render_options_digest, \
    render_types_digest
from predicators.agent_sdk.tools.program_synthesis import CandidateLoader, \
    create_program_synthesis_tools
from predicators.agent_sdk.tools.snapshots import finalize_versioned_snapshot
from predicators.approaches.agent_sim_learning_approach import _SynthesisPaths
from predicators.approaches.agent_sim_predicate_invention_approach import \
    AgentSimPredicateInventionApproach
from predicators.code_sim_learning.program_world_model import \
    ProgramOptionModel, ProgramWorldModel, load_program_world_model, \
    option_transitions, roll_program_latents
from predicators.settings import CFG
from predicators.structs import LowLevelTrajectory, State, Task

logger = logging.getLogger(__name__)


class AgentProgramWorldModelApproach(AgentSimPredicateInventionApproach):
    """Invented predicates + an option-level program world model."""

    _save_suffix: str = "AgentProgramWM"
    _CHECKPOINT_SANDBOX_FILES = (
        AgentSimPredicateInventionApproach._CHECKPOINT_SANDBOX_FILES +
        ("world_model.py", ))
    _CHECKPOINT_SANDBOX_DIRS = (
        AgentSimPredicateInventionApproach._CHECKPOINT_SANDBOX_DIRS +
        ("world_model_versions", ))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._program: Optional[ProgramWorldModel] = None
        self._program_model: Optional[ProgramOptionModel] = None
        # The belief particles ride the capture gate's rule-parameter
        # margin channel: every submission is re-rolled from each
        # particle's hidden state.
        ctx = self._tool_context
        ctx.rule_param_margin_provider = self._belief_particles
        ctx.rule_param_override_scope = self._particle_override_scope
        ctx.rule_param_margin_label = "belief particle"
        ctx.rule_param_margin_note = (
            "particles of the belief over the world model's hidden state")

    @classmethod
    def get_name(cls) -> str:
        return "agent_program_world_model"

    # ── Paths ────────────────────────────────────────────────────

    @staticmethod
    def _world_model_paths(paths: _SynthesisPaths) -> Dict[str, str]:
        """Host and agent-visible paths of the program artifact, mapped like
        the residual arm's simulator.py."""
        return {
            "world_model_file":
            os.path.join(paths.base, "world_model.py"),
            "versions_dir":
            os.path.join(paths.base, "world_model_versions"),
            "world_model_file_for_agent":
            paths.simulator_file_for_agent.replace("simulator.py",
                                                   "world_model.py"),
        }

    # ── Learning ─────────────────────────────────────────────────

    def _learn_simulator(self, trajectories: List[LowLevelTrajectory]) -> None:
        """Run one program-synthesis session and deploy what it wrote."""
        self._fit_trajectories = list(trajectories)
        self._persist_fit_trajectories("recorded")
        usable = [
            t for t in trajectories if t.actions and t.actions[0].has_option()
        ]
        if not usable and not CFG.agent_sim_learn_zero_shot:
            logger.warning("No skill-level transitions; skipping world "
                           "model synthesis.")
            return
        if not usable:
            logger.info("Zero-shot synthesis: no skill-level transitions; "
                        "the agent writes the world model without data.")
        program = self._run_program_synthesis_session(trajectories)
        if program is None:
            logger.warning("Synthesis produced no loadable world model; "
                           "the previous model stands.")
            return
        self._install_program(program)

    def _install_program(self, program: ProgramWorldModel) -> None:
        self._program = program
        self._program_model = ProgramOptionModel(program, seed=CFG.seed)
        self._option_model = self._program_model
        logger.info("Deployed the program world model (latent over %s).",
                    dict(program.latent_features) or "nothing")

    def _run_program_synthesis_session(
            self, trajectories: List[LowLevelTrajectory]
    ) -> Optional[ProgramWorldModel]:
        paths = self._resolve_synthesis_paths()
        wm_paths = self._world_model_paths(paths)
        extra_paths = self._compute_extra_synthesis_paths(paths.base)
        exec_ns = self._build_synthesis_exec_ns(trajectories)
        self._attach_program_session_state(exec_ns, trajectories, paths,
                                           wm_paths, extra_paths)
        # Fresh session so the synthesis prompt + tools take effect.
        self._close_agent_session()
        self._ensure_agent_session()
        structs_ref = self._write_structs_reference()
        message = self._build_program_learn_message(trajectories, paths,
                                                    wm_paths, structs_ref,
                                                    extra_paths)
        try:
            responses = self._query_agent_sync(message, kind="learn")
            dead = query_fatal_error(responses)
            if dead is not None:
                raise AgentSessionFatalError(
                    "The learn session died without the agent doing any "
                    f"work ({dead}); refusing to checkpoint this cycle as "
                    "learned.")
        finally:
            ctx = self._tool_context
            ctx.extra_session_hooks = {}
            ctx.extra_mcp_tools = []
            ctx.probe_artifact_loaders.clear()
            ctx.probe_option_model_provider = None
            ctx.probe_fit_provider = None
            ctx.probe_residuals_provider = None
            ctx.probe_score_provider = None
            ctx.probe_param_status = None
            ctx.learn_cycle_index = None
            self._learning_mode = False
            self._close_agent_session()
        return self._load_program_artifacts(wm_paths, extra_paths)

    def _attach_program_session_state(
        self,
        exec_ns: Dict[str, Any],
        trajectories: List[LowLevelTrajectory],
        paths: _SynthesisPaths,
        wm_paths: Dict[str, str],
        extra_paths: Dict[str, str],
    ) -> None:
        """Install this session's tools, probe, and snapshot hooks."""
        # pylint: disable=import-outside-toplevel
        from predicators.agent_sdk.belief_probe import _check_time_budget, \
            build_probe_namespace

        # pylint: enable=import-outside-toplevel
        ctx = self._tool_context
        ctx.learn_cycle_index = self._learning_cycle_index()
        toolkit = create_program_synthesis_tools(
            exec_ns,
            trajectories=trajectories,
            predicates=self._get_all_predicates(),
            types=self._types,
            options=self._get_all_options(),
            world_model_file=wm_paths["world_model_file"],
            versions_dir=wm_paths["versions_dir"],
            sandbox_dir=paths.base,
            sandbox_dir_for_agent=paths.sandbox_dir_for_agent,
            cycle_index_provider=self._learning_cycle_index,
            budget_check=lambda: _check_time_budget(ctx),
            rng=np.random.default_rng(CFG.seed),
        )
        self._install_extra_synthesis_surfaces(exec_ns, [], {}, extra_paths)
        declared = set(self._get_synthesis_tool_names() or ())
        ctx.extra_mcp_tools = [
            t for t in toolkit.tools if getattr(t, "name", "") in declared
        ]
        ctx.probe_option_model_provider = \
            self._make_candidate_program_provider(toolkit.load_candidate)
        ctx.probe_score_provider = toolkit.score_runner
        ctx.probe_fit_provider = None
        ctx.probe_residuals_provider = None
        probe_ns = build_probe_namespace(ctx)
        exec_ns["sim"] = probe_ns["sim"]
        exec_ns["BeliefProbe"] = probe_ns["BeliefProbe"]
        self._learning_mode = True
        targets = self._build_write_snapshot_targets(
            wm_paths["world_model_file"], wm_paths["versions_dir"],
            extra_paths)
        ctx.extra_session_hooks = self._build_synthesis_session_hooks(
            targets, paths.base)

    def _build_write_snapshot_targets(
        self,
        simulator_file: str,
        versions_dir: str,
        extra_paths: Dict[str, str],
    ) -> List[_SnapshotTarget]:
        """The world model and predicates.py, snapshotted on every write.

        ``simulator_file`` / ``versions_dir`` carry the world-model
        paths here (the caller is this class's session setup).
        """
        return [
            _SnapshotTarget(
                live_file=simulator_file,
                versions_dir=versions_dir,
                artifact_name="world_model",
                cycle_index_provider=self._learning_cycle_index,
            ),
            _SnapshotTarget(
                live_file=extra_paths["predicates_file"],
                versions_dir=extra_paths["predicates_versions_dir"],
                artifact_name="predicates",
                cycle_index_provider=self._learning_cycle_index,
            ),
        ]

    def _make_candidate_program_provider(
            self, load_candidate: CandidateLoader
    ) -> Callable[[], ProgramOptionModel]:
        """Lazy option-model builder over the CANDIDATE world_model.py.

        Rebuilt whenever the file's snapshot tag changes; raises (into
        the tool output) while no loadable candidate exists, so the
        probe never falls back to a pre-synthesis model.
        """
        cache: Dict[str, Any] = {}

        def _provider() -> ProgramOptionModel:
            program, tag, err = load_candidate(None)
            if err is not None:
                raise RuntimeError(
                    "run_python probe: no loadable candidate world model - "
                    f"{err}")
            assert program is not None
            if cache.get("tag") == tag:
                return cache["model"]
            model = ProgramOptionModel(program, seed=CFG.seed)
            self._tool_context.probe_param_status = (
                f"candidate world_model.py {tag}")
            cache["tag"] = tag
            cache["model"] = model
            logger.info("Synthesis probe: candidate world model rebuilt (%s).",
                        tag)
            return model

        return _provider

    def _build_program_learn_message(
        self,
        trajectories: List[LowLevelTrajectory],
        paths: _SynthesisPaths,
        wm_paths: Dict[str, str],
        structs_ref: str,
        extra_paths: Dict[str, str],
    ) -> str:
        predicates = self._get_all_predicates()
        n_trajs = len(trajectories)
        n_demos = sum(1 for t in trajectories if t.is_demo)
        n_transitions = sum(
            len(option_transitions(t, predicates)) for t in trajectories)
        prior: List[str] = []
        if os.path.isfile(wm_paths["world_model_file"]):
            prior.append("`./world_model.py`")
        if os.path.isfile(os.path.join(paths.base, "predicates.py")):
            prior.append("`./predicates.py`")
        session_tool_names = (self._agent_session.tool_names
                              if self._agent_session is not None else [])
        extra_messages: List[str] = []
        if not trajectories and CFG.agent_sim_learn_zero_shot:
            extra_messages.append(
                learn_prompts.render_program_zero_shot_message())
        extra_message = self._extra_synthesis_message(extra_paths)
        if extra_message:
            extra_messages.append(extra_message)
        return learn_prompts.build_program_learn_message(
            n_trajs=n_trajs,
            n_transitions=n_transitions,
            n_demos=n_demos,
            n_interaction=n_trajs - n_demos,
            trajectory_listing=self._format_trajectory_listing(trajectories),
            structs_ref=structs_ref,
            predicate_listing=self._format_predicate_signatures(predicates),
            types_digest=render_types_digest(self._tool_context.types),
            options_digest=render_options_digest(
                self._tool_context.options,
                gt_options_ref_path=self._tool_context.gt_options_ref_path),
            world_model_file=wm_paths["world_model_file_for_agent"],
            objective_block=self._format_objective_block(),
            prior_state_block=learn_prompts.render_prior_state_block(prior),
            tools_block=learn_prompts.render_tools_block(session_tool_names),
            extra_messages=extra_messages,
        )

    def _build_synthesis_system_prompt(self) -> str:
        return learn_prompts.build_program_learn_system_prompt(
            scene_viz_hint=self._scene_viz_hint(),
            extra_sections=self._extra_synthesis_system_prompt_sections(),
            workflow_extra=self._synthesis_workflow_extra(),
        )

    def _load_program_artifacts(
            self, wm_paths: Dict[str, str],
            extra_paths: Dict[str, str]) -> Optional[ProgramWorldModel]:
        """Load what the finished session committed to disk."""
        tag = finalize_versioned_snapshot(
            wm_paths["world_model_file"],
            wm_paths["versions_dir"],
            cycle_idx=self._learning_cycle_index(),
            artifact_name="world_model",
        )
        if tag is not None:
            # The trajectories' provenance stamp: the model version they
            # were collected under.
            self._current_simulator_version = tag
            logger.info("Final world model snapshot: %s", tag)
        program, err = self._load_program_file(wm_paths["world_model_file"])
        if program is None:
            logger.warning("world_model.py did not load: %s", err)
            return None
        self._post_synthesis_loading(extra_paths, [])
        return program

    def _load_program_file(
            self,
            path: str) -> Tuple[Optional[ProgramWorldModel], Optional[str]]:
        if not os.path.isfile(path):
            return None, f"no file at {path}"
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        return load_program_world_model(code, self._types,
                                        self._kept_initial_predicates,
                                        self._get_all_options())

    # ── Belief over the hidden state ─────────────────────────────

    def _attach_initial_latent(self, task: Task) -> Task:
        """Seed ``task.init.latent`` with the nominal particle (a seeded draw
        from the program's ``initial_latent``)."""
        if self._program_model is None:
            return task
        init_state = task.init.copy()
        init_state.latent = self._program_model.initial_latent(
            task.init, rng=np.random.default_rng(CFG.seed))
        return Task(init=init_state,
                    goal=task.goal,
                    alt_goal=task.alt_goal,
                    goal_nl=task.goal_nl)

    def _belief_particles(self) -> List[Dict[str, float]]:
        """Distinct draws from ``initial_latent`` for the current task: the
        capture gate's margin points (empty until a model exists)."""
        model = self._program_model
        if model is None:
            return []
        task = self._tool_context.current_task
        if task is None:
            if not self._train_tasks:
                return []
            task = self._train_tasks[0]
        rng = np.random.default_rng(CFG.seed + 7919)
        particles: List[Dict[str, Any]] = []
        seen = set()
        for _ in range(CFG.agent_program_belief_particles):
            try:
                latent = model.initial_latent(task.init, rng=rng)
            except utils.OptionExecutionFailure as e:
                logger.warning("Belief particles unavailable: %s", e)
                break
            key = repr(sorted(latent.items(), key=lambda kv: str(kv[0])))
            if key in seen:
                continue
            seen.add(key)
            particles.append(latent)
        return cast(List[Dict[str, float]], particles)

    @contextmanager
    def _particle_override_scope(self,
                                 particle: Dict[str, float]) -> Iterator[None]:
        """Roll every latent-less start from ``particle`` while entered."""
        model = self._program_model
        assert model is not None
        prev = model.initial_latent_override
        model.initial_latent_override = copy.deepcopy(particle)
        try:
            yield
        finally:
            model.initial_latent_override = prev

    def materialise_latent(
            self, traj: LowLevelTrajectory) -> List[Optional[Dict[str, Any]]]:
        if self._program is None:
            return [None] * len(traj.states)
        return roll_program_latents(self._program, traj,
                                    self._get_all_predicates(),
                                    np.random.default_rng(CFG.seed))

    def _latent_tracking_available(self) -> bool:
        # No execution-time tracker for the program's latent: episodes
        # run open-loop on the certified plan.
        return False

    # ── Env scope ────────────────────────────────────────────────

    @contextmanager
    def _fresh_validation_env_scope(
        self,
        physical_overrides: Optional[Dict[str,
                                          float]] = None) -> Iterator[None]:
        """No physics to refresh: the program is deterministic given its latent
        (which the particle scope varies), so validation repeats run on the
        model as is."""
        del physical_overrides
        yield

    # ── Checkpointing ────────────────────────────────────────────

    def _rehydrate_from_artifacts(self) -> None:
        paths = self._resolve_synthesis_paths()
        wm_paths = self._world_model_paths(paths)
        if not os.path.isfile(wm_paths["world_model_file"]):
            logger.info("Checkpoint carried no world_model.py; the initial "
                        "option model stands.")
            self._rehydrate_extra_artifacts(paths.base)
            return
        program, err = self._load_program_file(wm_paths["world_model_file"])
        if program is None:
            logger.warning(
                "Restored world_model.py failed to load (%s); continuing "
                "with the initial option model.", err)
            self._rehydrate_extra_artifacts(paths.base)
            return
        self._install_program(program)
        self._rehydrate_extra_artifacts(paths.base)
        logger.info(
            "Rehydrated the program world model from checkpoint artifacts "
            "(%d learned predicates).",
            len(getattr(self, "_learned_predicates", set()) or set()))


__all__ = ["AgentProgramWorldModelApproach", "State"]
