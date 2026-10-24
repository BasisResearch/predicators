"""Bilevel planning with an agent-written option-level program world model and
invented predicates (paper arm C4: a code world model with no engine
underneath, in the form of Pinductor / POMDP Coder).

Everything about the loop is the residual arm's - the sketch / refine /
run tools and predicate invention - except the model artifact: instead of
residual rules over a physics engine the agent writes ``world_model.py``,
an option-level transition program with its own hidden state (see
:mod:`code_sim_learning.program_world_model`). There is no parameter
fit; the synthesis tools score the program with the Pinductor
particle-filter kernel pseudo-likelihood (``sim.score``) and the agent
edits it.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, \
    Tuple

import numpy as np

from predicators.agent_sdk.tools import _SnapshotTarget
from predicators.agent_sdk.tools.program_synthesis import CandidateLoader, \
    create_program_synthesis_tools
from predicators.agent_sdk.tools.snapshots import finalize_versioned_snapshot
from predicators.approaches.agent_sim_learning_approach import _SynthesisPaths
from predicators.approaches.agent_sim_predicate_invention_approach import \
    AgentSimPredicateInventionApproach
from predicators.code_sim_learning.program_world_model import \
    ProgramOptionModel, ProgramWorldModel, load_program_world_model, \
    roll_program_latents
from predicators.settings import CFG
from predicators.structs import LowLevelTrajectory, State

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

    def _install_program(self, program: ProgramWorldModel) -> None:
        self._program = program
        self._program_model = ProgramOptionModel(program, seed=CFG.seed)
        self._option_model = self._program_model
        logger.info("Deployed the program world model (latent over %s).",
                    dict(program.latent_features) or "nothing")

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
            **self._program_tool_overrides(),
        )
        ctx.probe_disabled = self._program_probe_disabled()
        self._install_extra_synthesis_surfaces(exec_ns, [], {}, extra_paths)
        declared = set(self._get_synthesis_tool_names() or ())
        ctx.extra_mcp_tools = [
            t for t in toolkit.tools if getattr(t, "name", "") in declared
        ]
        ctx.probe_option_model_provider = \
            self._make_candidate_program_provider(toolkit.load_candidate)
        ctx.probe_score_provider = toolkit.score_runner
        ctx.probe_fit_provider = None
        ctx.probe_validation_provider = None
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

    def _program_tool_overrides(self) -> Dict[str, Any]:
        """Keyword overrides for the program-synthesis toolkit (none here; an
        arm with a narrower probe supplies its own ``run_python``
        description)."""
        return {}

    def _program_probe_disabled(self) -> FrozenSet[str]:
        """The ``sim`` calls this approach's probe refuses (none here)."""
        return frozenset()

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
