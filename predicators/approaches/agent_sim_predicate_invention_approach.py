"""Agent sim-learning + predicate-invention approach.

Extends ``AgentSimLearningApproach`` so the synthesizing Claude agent
can also invent the symbolic predicates used for plan subgoals. The
env's predicates are stripped down to a primitive allowlist (default:
``{"Holding"}``), and the agent is asked to define
``LEARNED_PREDICATES`` in a sandboxed ``predicates.py``. The invented
predicates flow through ``_get_all_predicates`` so they are visible to
backtracking refinement, the option model's abstraction function, and
every other call site that asks the approach for its current
predicates.

Predicates persist across online learning cycles — ``predicates.py``
is preserved at the sandbox root, and every version evaluated during
synthesis (plus a final snapshot of any post-eval edits) is saved to
``predicates_versions/`` as ``cycle_XXX_vers_YYY_predicates.py``.

Example command::

    python predicators/main.py --env pybullet_boil \
        --approach agent_sim_predicate_invention --seed 0 \
        --num_train_tasks 10 --num_test_tasks 5 \
        --num_online_learning_cycles 2 --explorer agent_plan
"""

import logging
import os
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from predicators.agent_sdk.tools import PREDICATE_SYNTHESIS_TOOL_NAMES, \
    SCENE_TOOL_NAMES, _SnapshotTarget, create_predicate_synthesis_tools, \
    finalize_versioned_snapshot
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.settings import CFG
from predicators.structs import Action, DerivedPredicate, Predicate, State

logger = logging.getLogger(__name__)


class AgentSimPredicateInventionApproach(AgentSimLearningApproach):
    """Bilevel planning with learned simulator AND invented predicates.

    See module docstring.
    """

    KEPT_INITIAL_PREDICATE_NAMES: FrozenSet[str] = frozenset({"Holding"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._learned_predicates: Set[Predicate] = set()
        self._kept_initial_predicates: Set[Predicate] = (
            self._compute_kept_initial_predicates())
        # We hide env goal predicate atoms from the agent and only present
        # goals as natural language; the env therefore owes us a goal_nl
        # for every train task.
        missing = [i for i, t in enumerate(self._train_tasks) if not t.goal_nl]
        assert not missing, (
            f"{type(self).__name__} requires every train task to set "
            f"`goal_nl` (env goal atoms are deliberately not exposed to "
            f"the agent). Missing on task indices: {missing}")
        kept_names = sorted(p.name for p in self._kept_initial_predicates)
        stripped = sorted(p.name for p in self._initial_predicates
                          if p not in self._kept_initial_predicates)
        logger.info(
            "Predicate stripping: kept %s; stripped (must be invented): %s",
            kept_names, stripped)

    @classmethod
    def get_name(cls) -> str:
        return "agent_sim_predicate_invention"

    # ── Predicate set ───────────────────────────────────────────

    def _get_all_predicates(self) -> Set[Predicate]:
        return self._kept_initial_predicates | self._learned_predicates

    def _compute_kept_initial_predicates(self) -> Set[Predicate]:
        """Apply the allowlist + closure-strip on derived predicates.

        A ``DerivedPredicate`` whose ``auxiliary_predicates`` references
        any stripped predicate is itself stripped — keeping a derived
        predicate whose dependencies have been removed would expose a
        broken classifier to refinement.
        """
        kept_names = self._resolve_kept_names()
        kept = {p for p in self._initial_predicates if p.name in kept_names}
        kept_pred_set = set(kept)
        for pred in self._initial_predicates:
            if not isinstance(pred, DerivedPredicate):
                continue
            if pred in kept_pred_set:
                aux = pred.auxiliary_predicates or set()
                if any(a not in kept_pred_set for a in aux):
                    kept.discard(pred)
        return kept

    def _resolve_kept_names(self) -> FrozenSet[str]:
        cfg_override = getattr(
            CFG, "agent_sim_predicate_invention_kept_predicate_names", None)
        if cfg_override:
            return frozenset(cfg_override)
        return self.KEPT_INITIAL_PREDICATE_NAMES

    # ── Agent session hooks ─────────────────────────────────────

    def _get_solve_tool_names(self) -> Optional[List[str]]:
        """Extend the planner's tool subset with the SCENE tools.

        ``annotate_scene`` and ``visualize_state`` are useful for
        predicate invention: rendering the scene lets the agent confirm
        geometry it would otherwise have to infer numerically. The
        parent (``AgentPlannerApproach``) gates these on
        ``agent_planner_use_*`` CFG flags, but those names refer to a
        different use case — for predicate invention we always want them
        available.
        """
        names = super()._get_solve_tool_names()
        if names is None:
            return None
        for extra in SCENE_TOOL_NAMES:
            if extra not in names:
                names.append(extra)
        return names

    def _get_synthesis_tool_names(self) -> Optional[List[str]]:
        """Extend the sim-learning synthesis surface with SCENE tools and the
        predicate-synthesis callable.

        Adds ``visualize_state`` / ``annotate_scene`` (the
        predicate-invention prompt explicitly tells the agent to call
        them when verifying geometric thresholds) and
        ``evaluate_predicate_quality`` (the dynamic tool built by
        :meth:`_extra_synthesis_tools`).
        """
        names = super()._get_synthesis_tool_names()
        if names is None:
            return None
        for extra in list(SCENE_TOOL_NAMES) + list(
                PREDICATE_SYNTHESIS_TOOL_NAMES):
            if extra not in names:
                names.append(extra)
        return names

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

    def _extra_synthesis_tools(
        self,
        exec_ns: Dict[str, Any],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
        extra_paths: Dict[str, str],
    ) -> List[Any]:
        del exec_ns, base_pred_triples, inferred_hint
        trajectories = self._get_all_trajectories()
        return create_predicate_synthesis_tools(
            predicates_file=extra_paths["predicates_file"],
            predicates_versions_dir=extra_paths["predicates_versions_dir"],
            approach=self,
            trajectories=trajectories,
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
        path = extra_paths["predicates_file_for_agent"]
        goal_block = self._format_goal_nl_block()
        return f"""\
## Predicate Invention

Important: this approach has stripped the env's symbolic predicates down \
to the "## Available Predicates" allowlist above (just `Holding` by \
default). You must invent everything else used as a subgoal in plan \
sketches — placements (e.g. JugAtFaucet), device states (FaucetOn / \
FaucetOff), and process completions (e.g. WaterBoiled) — by writing them \
to `{path}` as `LEARNED_PREDICATES`. See the system prompt section \
"Predicate Invention" for the file format.

{goal_block}\
Goal achievement is checked externally — the env owns the goal \
definition. You do **not** need to invent goal predicates or match any \
env predicate names. To check whether a state satisfies the goal, call \
the black-box reward `is_goal_state(state, task_idx)` (equivalently \
`train_tasks[task_idx].goal_holds(state)`). Refinement uses the same \
env-side check, so your invented predicates are free to use any names \
you like and only need to support plan-sketch subgoals (gating Wait, \
Place, etc.).

Failure trajectories are signal: when an interaction trajectory has \
`reached_goal=False`, look for points where your predicate was true but \
downstream progress stalled (e.g. a placement predicate fires but the \
relevant rule feature stops advancing). That's evidence the threshold \
is too loose; tighten it or share the gating parameter with the rule \
via `params[...]` so MCMC can fit them jointly.

Workflow: edit `predicates.py`, call `evaluate_predicate_quality` \
(fast, also reloads predicates into the live set), then call \
`evaluate_plan_refinement` with sketches that reference your invented \
names. Any predicate you reference in a sketch must exist in \
`predicates.py` first."""

    def _format_goal_nl_block(self) -> str:
        """Render the natural-language goals for the train tasks.

        Lists each task's `goal_nl`, deduped (since several tasks often
        share the same goal description). Returns an empty string only
        if every task is missing one — but ``__init__`` asserts they're
        present, so in practice this always returns a non-empty block.
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

    def _extra_synthesis_system_prompt(self) -> str:
        return _PREDICATE_PROMPT_SECTION

    def _post_synthesis_loading(
        self,
        extra_paths: Dict[str, str],
        specs: List[Any],
    ) -> None:
        """Load predicates.py and snapshot the cycle's final state."""
        predicates_file = extra_paths["predicates_file"]
        predicates_versions_dir = extra_paths["predicates_versions_dir"]

        # Seed _fitted_params from init values so predicate lambdas
        # closing over ``params["..."]`` can be evaluated during
        # validation. The actual MCMC fit runs later in the base flow
        # and will overwrite these values. Mutate in place so
        # _ParamsView holders pick up the seeds.
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
        from predicators.agent_sdk.proposal_parser import build_exec_context, \
            exec_code_safely, validate_predicate
        from predicators.agent_sdk.tools import _ParamsView
        from predicators.code_sim_learning.training import ParamSpec

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


_PREDICATE_PROMPT_SECTION = """\
## Predicate Invention (required for plan subgoals)

You are responsible for inventing the symbolic predicates the planner \
will use as subgoal atoms in plan sketches. Only `Holding` is provided \
as a primitive; placement, device-state, and process-completion \
predicates do not exist until you invent them.

Goals are presented to you in natural language (see the synthesis \
message). Goal achievement is checked externally by the env via \
`is_goal_state(state, task_idx)` / `train_tasks[task_idx].goal_holds(state)`. \
You do **not** need to invent any goal-named predicates and you do \
**not** need to match env predicate names. Your invented predicates \
are purely for plan-sketch subgoals (gating Wait/Place/etc.) and can \
be named freely.

Define them in `predicates.py` (path given in the first message):

```python
LEARNED_PREDICATES: List[Predicate]
```

The exec namespace pre-injects `Predicate` and a `<typename>_type` binding \
for each env type (e.g. `jug_type`, `faucet_type`). Example:

```python
LEARNED_PREDICATES = [
    Predicate("JugAtFaucet", [jug_type, faucet_type],
              lambda s, objs: ((s.get(objs[0], "x") - s.get(objs[1], "x"))**2
                               + (s.get(objs[0], "y") - s.get(objs[1], "y"))**2)
                              < params["jug_at_faucet_dist"]**2),
    Predicate("FaucetOn", [faucet_type],
              lambda s, objs: s.get(objs[0], "is_on") > 0.5),
    Predicate("BoilingDone", [jug_type],
              lambda s, objs: s.get(objs[0], "heat_level") >= params["boiled_threshold"]),
]
```

A pre-injected `params` view is in scope; it always reads the **current \
fitted values** of every `ParamSpec` declared in `simulator.py`. Whenever \
MCMC re-fits, predicates picking up `params["name"]` see the new values \
automatically. To share a threshold between a rule and a predicate, declare \
it once in `PARAM_SPECS` and reference `params["name"]` from both — this \
is the recommended pattern when a single physical threshold gates both \
process dynamics (the rule's "fire" condition) and a control-relevant \
predicate (the planner's "this subgoal is reached" check).

Caveat: a parameter used only by predicates (not by any rule) has no SSE \
signal — it stays at `init_value`. Pick good initial values for those.

What you'll need (typical pattern):
- Placement predicates (object at a target location) for any open-ended \
option like Place — refinement needs these or it picks an arbitrary location.
- Device-state predicates (on/off) for any toggle option.
- Process-completion predicates over the features your rules drive, so \
Wait steps know when to terminate. Keep classifier thresholds consistent \
with rule saturation values; an inconsistency causes evaluate_step_fit to \
look fine while evaluate_plan_refinement gets stuck on the Wait subgoal.

Verifying classifiers against the scene and data (applies to all predicates):

A classifier picks features and parameter values; both can be wrong. Do \
not pick either from intuition — verify before committing.

- `visualize_state` / `annotate_scene` (available for any PyBullet env): \
use whenever a predicate depends on geometry. A body's recorded pose \
often doesn't coincide with the feature that matters (a faucet's spout, \
a switch's handle, a burner's hot zone, the inside of a container); \
render the scene, annotate candidate target points / regions, and \
confirm what's actually where before encoding a threshold.
- `run_python` (numerical workbench): iterate trajectory states and \
compute the candidate classifier (or its underlying numeric expression) \
at each step. The right parameter values cleanly separate the steps \
where a downstream effect actually happens — the relevant rule feature \
advances, the goal-relevant quantity changes — from the steps where it \
doesn't. Sweep candidates against that signal and pick by separation. \
This applies to every kind of predicate: placement thresholds, \
process-completion cutoffs, on/off comparison points, etc.

Validate with `evaluate_predicate_quality` (cheap; reports first-flip step, \
monotonicity, coverage across all available trajectories). On goal-reaching \
trajectories (`reached_goal=True` in `inspect_trajectories`) a milestone \
predicate should flip False→True exactly once and stay true; on failed \
interaction trajectories (`reached_goal=False`) the same predicate may \
fire but the rest of the trajectory won't show goal completion — useful \
signal for spotting an over-loose threshold (predicate fires, downstream \
physics doesn't follow). A placement predicate should be true exactly \
when an object is at its intended location and false otherwise.

`evaluate_predicate_quality` is also the loader: it updates the predicate \
set used by `evaluate_plan_refinement`. Call it after every edit to \
`predicates.py` before re-running plan refinement.

Predicates persist across online cycles — the file is preserved between \
synthesis sessions. Edit it freely; every successful Write/Edit (and a \
final post-session check) is snapshotted to \
`predicates_versions/cycle_XXX_vers_YYY_predicates.py`. Each online cycle \
re-runs synthesis with the full trajectory history (offline demos + every \
interaction trajectory collected so far), so failed past attempts remain \
visible for the agent to learn from.
"""
