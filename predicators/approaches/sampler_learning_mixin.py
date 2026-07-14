"""Mixin providing per-skill sampler learning for the sim-learning approach.

Samplers are a first-class artifact of the base sim-learning approach
(gated by ``CFG.agent_sim_learn_synthesize_samplers``), not a subclass
extension like predicates — so they are woven into
``AgentSimLearningApproach._synthesize_with_agent`` and
``_learn_simulator`` directly rather than via the ``_extra_synthesis_*``
hooks, which keeps them independent of the predicate subclass's
(non-super-calling) hook overrides. When a sim-synthesis session runs
(``oracle_sim_program=False``) the sampler tool/snapshot/message ride
along in it; when none runs (``oracle_sim_program=True``) they get a
dedicated session via :meth:`_synthesize_samplers_standalone`.

This mixin owns everything sampler-specific: mode resolution (learn vs.
ground truth), sandbox path bindings, the synthesis tool/snapshot/message
builders, loading ``LEARNED_SAMPLERS`` from file, and the standalone
synthesis session. The host approach keeps only the call sites.
"""
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import numpy as np

from predicators.agent_sdk.tools import _SnapshotTarget, \
    create_sampler_synthesis_tools, create_synthesis_tools, \
    finalize_versioned_snapshot
from predicators.code_sim_learning.training import ParamSpec
from predicators.ground_truth_models import get_gt_samplers
from predicators.settings import CFG
from predicators.structs import Action, LowLevelTrajectory, OptionSampler, \
    ParameterizedOption, Predicate, State, Task, Type

if TYPE_CHECKING:
    from predicators.agent_sdk.tools import ToolContext

logger = logging.getLogger(__name__)


class SamplerLearningMixin:
    """Per-skill sampler synthesis, loading, and oracle installation.

    Mixed into :class:`AgentSimLearningApproach`. Holds the
    sampler-learning state (``_do_synthesize_samplers``,
    ``_current_samplers_version``) — the host ``__init__`` must call
    :meth:`_init_sampler_learning_state`.
    """

    # ── Host-class contract ─────────────────────────────────────
    # Everything below is provided by the host approach (its
    # AgentSessionMixin / BaseApproach ancestry or the host class
    # itself). Declared under TYPE_CHECKING only, so these never
    # shadow the real implementations in the MRO at runtime.
    if TYPE_CHECKING:
        _tool_context: "ToolContext"
        _train_tasks: List[Task]
        _types: Set[Type]
        _fitted_params: Dict[str, float]
        _learning_mode: bool
        _synthesized_samplers: Dict[str, OptionSampler]

        def _learning_cycle_index(self) -> int:
            raise NotImplementedError

        def _get_log_dir(self) -> str:
            raise NotImplementedError

        def _get_all_predicates(self) -> Set[Predicate]:
            raise NotImplementedError

        def _get_all_options(self) -> Set[ParameterizedOption]:
            raise NotImplementedError

        def _get_synthesis_tool_names(self) -> Optional[List[str]]:
            raise NotImplementedError

        def _query_agent_sync(self, message: str,
                              **query_kwargs: Any) -> List[Dict[str, Any]]:
            raise NotImplementedError

        def _ensure_agent_session(self) -> None:
            raise NotImplementedError

        def _close_agent_session(self) -> None:
            raise NotImplementedError

        @staticmethod
        def _build_synthesis_session_hooks(
                targets: List[_SnapshotTarget],
                sandbox_dir: str) -> Dict[str, list]:
            raise NotImplementedError

        @staticmethod
        def _format_predicate_signatures(predicates: Set[Predicate]) -> str:
            raise NotImplementedError

    def _init_sampler_learning_state(self) -> None:
        """Initialize sampler state; called from the host ``__init__``."""
        # Snapshot tag of the most recent samplers file committed by the
        # synthesis agent — used to stamp newly collected online
        # trajectories with their source-version provenance.
        self._current_samplers_version: Optional[str] = None
        # Whether this run learns samplers (vs. using ground-truth ones).
        # Refined per cycle in _learn_simulator once GT availability is
        # known; this default is what the synthesis-session tool surface
        # reads.
        self._do_synthesize_samplers: bool = (
            CFG.agent_sim_learn_synthesize_samplers
            and not CFG.agent_sim_learn_oracle_samplers)

    @staticmethod
    def _samplers_enabled() -> bool:
        """Whether per-skill samplers are used at all this run."""
        return CFG.agent_sim_learn_synthesize_samplers

    def _maybe_install_oracle_samplers(self) -> None:
        """Resolve sampler mode for this cycle and install GT ones if used.

        Sets ``self._do_synthesize_samplers`` (learn vs. use ground
        truth). When ``agent_sim_learn_oracle_samplers`` is on and the
        env provides ground-truth samplers, installs them and skips
        synthesis; if none exist, warns and falls back to synthesis.
        """
        gt_samplers = None
        if self._samplers_enabled() and CFG.agent_sim_learn_oracle_samplers:
            gt_samplers = get_gt_samplers(CFG.env)
            if gt_samplers:
                self._synthesized_samplers = dict(gt_samplers)
                self._current_samplers_version = "oracle"
                logger.info("Using %d ground-truth sampler(s): %s",
                            len(gt_samplers), ", ".join(sorted(gt_samplers)))
            else:
                logger.warning(
                    "agent_sim_learn_oracle_samplers=True but no ground-truth "
                    "samplers for env %s; falling back to synthesis.", CFG.env)
        self._do_synthesize_samplers = (self._samplers_enabled()
                                        and not gt_samplers)

    def _sampler_paths(self, base: str) -> Dict[str, str]:
        """Sandbox path bindings for samplers.py (host + agent-visible)."""
        samplers_file = os.path.join(base, "samplers.py")
        samplers_versions_dir = os.path.join(base, "samplers_versions")
        if CFG.agent_sdk_use_local_sandbox:
            samplers_file_for_agent = "./samplers.py"
        elif self._tool_context.sandbox_dir:
            samplers_file_for_agent = "/sandbox/samplers.py"
        else:
            samplers_file_for_agent = samplers_file
        return {
            "samplers_file": samplers_file,
            "samplers_versions_dir": samplers_versions_dir,
            "samplers_file_for_agent": samplers_file_for_agent,
        }

    def _make_sampler_tools(self, paths: Dict[str, str]) -> List[Any]:
        """Build the evaluate_sampler MCP tool for a synthesis session."""
        return create_sampler_synthesis_tools(
            samplers_file=paths["samplers_file"],
            samplers_versions_dir=paths["samplers_versions_dir"],
            approach=self,
            cycle_index_provider=self._learning_cycle_index,
        )

    def _sampler_snapshot_target(self, paths: Dict[str,
                                                   str]) -> _SnapshotTarget:
        """Snapshot target that versions samplers.py on every Write/Edit."""
        return _SnapshotTarget(
            live_file=paths["samplers_file"],
            versions_dir=paths["samplers_versions_dir"],
            artifact_name="samplers",
            cycle_index_provider=self._learning_cycle_index,
        )

    def _sampler_synthesis_message(self, paths: Dict[str, str]) -> str:
        """Instructions appended to the agent's first synthesis message."""
        path = paths["samplers_file_for_agent"]
        return f"""\
## Per-Skill Sampler Synthesis

Backtracking refinement draws each option's continuous parameters \
*uniformly* from its params box by default. When a sketch step's subgoal \
pins the parameters into a tiny region (e.g. a placement that must land \
within a few cm of an exact point and at a specific orientation), uniform \
sampling almost never hits it and refinement exhausts its budget. Fix this \
by writing per-skill samplers to `{path}` as a dict \
`LEARNED_SAMPLERS = {{"OptionName": sampler_fn, ...}}` keyed by option name.

Each sampler has signature \
`fn(state, subgoal_atoms, rng, objects) -> params` (the same signature as \
the env's NSRT samplers) where:
- `state` is the current `State` (read object features with `state.get(obj, "feat")`),
- `subgoal_atoms` is the set of `GroundAtom`s the step must establish — \
read the target relation here (e.g. an `InFront`/at-target atom names the \
two objects whose geometry the placement must satisfy) and compute the \
parameters that achieve it. At steps with NO subgoal annotation this set \
is EMPTY — the sampler must not crash on `set()`; fall back to a default \
or uniform draw,
- `rng` is a `numpy` `Generator` (use it for small jitter so retries differ),
- `objects` is the list of typed objects bound to this option call.
Return a `float32` array whose length matches the option's params box \
(see `inspect_options` for the dimension and ranges); refinement clips it \
to that box, so stay within the ranges.

Samplers are the reusable cross-task prior: refinement uses yours on \
every draw of that option, in every sketch and every task. A sketch step \
that carries its own `~ [widths]` region annotation bypasses the sampler \
for that step (precedence: per-step region > per-skill sampler > uniform).

Aim the parameters at the subgoal geometrically (then add a little `rng` \
jitter); do NOT just return uniform draws. Read the option signatures with \
`inspect_options` and the predicate classifiers (for the subgoal geometry) \
with the predicate listing above.

Workflow: write `{path}`, call `evaluate_sampler` (snapshots + installs \
them and sanity-checks shape/box), then call `evaluate_plan_refinement` \
with a sketch using those options — the samples-to-refine count should \
drop sharply versus uniform. Iterate with `Edit` and re-run. Every \
successful Write/Edit of `{path}` is snapshotted to `samplers_versions/` \
as `cycle_XXX_vers_YYY_samplers.py`."""

    def _finalize_and_load_samplers(self, paths: Dict[str, str]) -> None:
        """Snapshot the final samplers.py and load it into approach state."""
        tag = finalize_versioned_snapshot(
            paths["samplers_file"],
            paths["samplers_versions_dir"],
            cycle_idx=self._learning_cycle_index(),
            artifact_name="samplers",
        )
        if tag is not None:
            self._current_samplers_version = tag
            logger.info("Final samplers snapshot: %s", tag)
        loaded = self._load_samplers_from_module_file(paths["samplers_file"])
        self._synthesized_samplers = loaded
        logger.info("Loaded %d per-skill sampler(s) from %s.", len(loaded),
                    paths["samplers_file"])
        for name in sorted(loaded):
            logger.info("  sampler: %s", name)

    def _load_samplers_from_module_file(self,
                                        path: str) -> Dict[str, OptionSampler]:
        """Load LEARNED_SAMPLERS from ``path``; validate each entry.

        Mirrors ``_load_predicates_from_module_file``. Returns an empty
        dict on missing file or exec failure (samplers are optional).
        Validation (unknown option names, non-callables) is shared with
        the ``evaluate_sampler`` tool via ``load_learned_samplers``.
        """
        # pylint: disable=import-outside-toplevel
        from predicators.agent_sdk.proposal_parser import build_exec_context, \
            load_learned_samplers
        from predicators.agent_sdk.tools import _ParamsView

        # pylint: enable=import-outside-toplevel
        # ParamSpec is imported at module scope (used by exec'd samplers
        # that close over learned params, mirroring the predicate loader).

        if not os.path.isfile(path):
            logger.info("No samplers file at %s; sampler set is empty.", path)
            return {}

        with open(path, "r", encoding="utf-8") as f:
            code = f.read()

        ctx = build_exec_context(types=self._types,
                                 predicates=self._get_all_predicates(),
                                 options=self._get_all_options(),
                                 extra_context={
                                     "params":
                                     _ParamsView(self._fitted_params),
                                     "ParamSpec": ParamSpec,
                                 })

        option_names = {o.name for o in self._get_all_options()}
        valid, warnings, err = load_learned_samplers(code, ctx, option_names)
        if err is not None:
            logger.warning("Failed to load %s:\n%s", path, err)
            return {}
        for warning in warnings:
            logger.warning("%s: %s", path, warning)
        return valid

    def _synthesize_samplers_standalone(
            self, trajectories: List[LowLevelTrajectory],
            base_pred_triples: List[Tuple[State, Action, State]],
            inferred_hint: Dict[str, List[str]]) -> None:
        """Run a dedicated sampler-synthesis session.

        Used when oracle_sim_program short-circuits the sim-synthesis
        session, so samplers still get learned. Reuses that session's
        sandbox/snapshot/tool machinery. Called from _learn_simulator
        after the option model is built, so evaluate_plan_refinement has
        a working simulator.
        """
        if CFG.agent_sdk_use_local_sandbox:
            sandbox_dir: Optional[str] = os.path.abspath(
                os.path.join(self._get_log_dir(), "sandbox"))
        else:
            sandbox_dir = self._tool_context.sandbox_dir
        base = sandbox_dir or self._get_log_dir()

        if CFG.agent_sdk_use_local_sandbox:
            sandbox_dir_for_agent: Optional[str] = "."
        elif sandbox_dir:
            sandbox_dir_for_agent = "/sandbox"
        else:
            sandbox_dir_for_agent = None

        paths = self._sampler_paths(base)
        simulator_file = os.path.join(base, "simulator.py")
        versions_dir = os.path.join(base, "simulator_versions")

        exec_ns: Dict[str, Any] = {
            "trajectories":
            trajectories,
            "train_tasks":
            self._train_tasks,
            "is_goal_state":
            lambda state, task_idx: self._train_tasks[task_idx].goal_holds(
                state),
            "np":
            np,
            "ParamSpec":
            ParamSpec,
        }
        # evaluate_plan_refinement (from the standard synthesis tools) gives
        # the agent the samples-to-refine feedback signal; the sampler tool
        # installs + sanity-checks the samplers.
        tools = create_synthesis_tools(
            exec_ns,
            base_pred_triples,
            inferred_hint,
            simulator_file=simulator_file,
            versions_dir=versions_dir,
            approach=self,
            sandbox_dir=base,
            sandbox_dir_for_agent=sandbox_dir_for_agent,
            cycle_index_provider=self._learning_cycle_index,
        )
        tools.extend(self._make_sampler_tools(paths))
        # Use the same declared surface as the mixin will assert against
        # (_get_synthesis_tool_names already includes the sampler tool since
        # _do_synthesize_samplers is True here). The rule-fitting tools are
        # exposed but irrelevant — the message steers the agent to samplers.
        declared = set(self._get_synthesis_tool_names() or ())
        self._tool_context.extra_mcp_tools = [
            t for t in tools if getattr(t, "name", "") in declared
        ]
        self._learning_mode = True
        self._tool_context.extra_session_hooks = (
            self._build_synthesis_session_hooks(
                [self._sampler_snapshot_target(paths)], base))

        self._close_agent_session()
        self._ensure_agent_session()

        predicate_listing = self._format_predicate_signatures(
            self._get_all_predicates())
        message = f"""\
Synthesize per-skill samplers for this environment's options. The \
simulator dynamics are already fixed (oracle/learned); your only job is \
to make backtracking refinement land each option's continuous parameters \
on its sketch-step subgoal instead of drawing them uniformly.

## Available Predicates (subgoal geometry)
{predicate_listing}

Read the option signatures with `inspect_options` and explore the \
trajectory data with `run_python` (variables: `trajectories`, \
`train_tasks`, `is_goal_state`, `np`, `ParamSpec`)."""
        message = message + "\n\n" + self._sampler_synthesis_message(paths)

        try:
            self._query_agent_sync(message, kind="learn")
        finally:
            self._tool_context.extra_session_hooks = {}
            self._tool_context.extra_mcp_tools = []
            self._learning_mode = False
            self._close_agent_session()

        self._finalize_and_load_samplers(paths)
