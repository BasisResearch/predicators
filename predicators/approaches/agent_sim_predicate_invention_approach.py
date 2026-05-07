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
is preserved at the sandbox root, and each cycle's final state is
archived to ``predicates_archive/cycle_NNN_predicates.py``.

Example command::

    python predicators/main.py --env pybullet_boil \
        --approach agent_sim_predicate_invention --seed 0 \
        --num_train_tasks 10 --num_test_tasks 5 \
        --num_online_learning_cycles 2 --explorer agent_plan
"""

import logging
import os
import shutil
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from predicators.agent_sdk.tools import create_predicate_synthesis_tools
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
        self._predicates_cycle_count: int = 0
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

    # ── Synthesis hooks ──────────────────────────────────────────

    def _compute_extra_synthesis_paths(self,
                                       base: str) -> Dict[str, str]:
        predicates_file = os.path.join(base, "predicates.py")
        predicates_versions_dir = os.path.join(base, "predicates_versions")
        predicates_archive_dir = os.path.join(base, "predicates_archive")

        if CFG.agent_sdk_use_local_sandbox:
            predicates_file_for_agent = "./predicates.py"
        elif self._tool_context.sandbox_dir:
            predicates_file_for_agent = "/sandbox/predicates.py"
        else:
            predicates_file_for_agent = predicates_file

        return {
            "predicates_file": predicates_file,
            "predicates_versions_dir": predicates_versions_dir,
            "predicates_archive_dir": predicates_archive_dir,
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
        )

    def _extra_synthesis_message(self, extra_paths: Dict[str, str]) -> str:
        path = extra_paths["predicates_file_for_agent"]
        goal_sigs = self._format_goal_predicate_signatures()
        if goal_sigs:
            goal_block = (
                f"Goal predicates (these must be invented or refinement "
                f"can't check goal achievement):\n{goal_sigs}\n\n")
        else:
            goal_block = ""
        return (
            f"## Predicate Invention\n\n"
            f"Important: this approach has stripped the env's symbolic "
            f"predicates down to the \"## Available Predicates\" allowlist "
            f"above (just `Holding` by default). You must invent everything "
            f"else used as a subgoal in plan sketches — placements (e.g. "
            f"JugAtFaucet), device states (FaucetOn / FaucetOff), and "
            f"process completions (e.g. WaterBoiled) — by writing them to "
            f"`{path}` as `LEARNED_PREDICATES`. See the system prompt "
            f"section \"Predicate Invention\" for the file format.\n\n"
            f"{goal_block}"
            f"Goal expressibility: training-task goals reference the env's "
            f"original predicate names. For goals to remain checkable, "
            f"reuse those exact names with matching arity/types when you "
            f"invent the corresponding classifiers (a `WaterBoiled(jug)` "
            f"you invent will be treated as the same predicate as the "
            f"env's `WaterBoiled(jug)` — equality is by name+types). You "
            f"may also invent extra predicates with new names.\n\n"
            f"Workflow: edit `predicates.py`, call "
            f"`evaluate_predicate_quality` (fast, also reloads predicates "
            f"into the live set), then call `evaluate_plan_refinement` "
            f"with sketches that reference your invented names. Any "
            f"predicate you reference in a sketch must exist in "
            f"`predicates.py` first.")

    def _format_goal_predicate_signatures(self) -> str:
        """List `Name(t1, t2)` for every predicate used in any train goal.

        Restricted to predicates NOT in the kept allowlist (those still
        come from the env). Empty string if no goals reference stripped
        predicates.
        """
        kept_names = {p.name for p in self._kept_initial_predicates}
        goal_preds: Dict[str, Tuple[str, ...]] = {}
        for task in self._train_tasks:
            for atom in task.goal:
                if atom.predicate.name in kept_names:
                    continue
                sig = tuple(t.name for t in atom.predicate.types)
                goal_preds[atom.predicate.name] = sig
        if not goal_preds:
            return ""
        lines = []
        for name in sorted(goal_preds):
            lines.append(f"  {name}({', '.join(goal_preds[name])})")
        return "\n".join(lines)

    def _extra_synthesis_system_prompt(self) -> str:
        return _PREDICATE_PROMPT_SECTION

    def _post_synthesis_loading(
        self,
        extra_paths: Dict[str, str],
        specs: List[Any],
    ) -> None:
        """Load predicates.py and archive the cycle's final state."""
        predicates_file = extra_paths["predicates_file"]
        archive_dir = extra_paths["predicates_archive_dir"]

        # Seed _fitted_params from init values so predicate lambdas
        # closing over ``params["..."]`` can be evaluated during
        # validation. The actual MCMC fit runs later in the base flow
        # and will overwrite these values.
        if specs:
            self._fitted_params = {s.name: s.init_value for s in specs}

        loaded = self._load_predicates_from_module_file(predicates_file)
        self._learned_predicates = loaded
        logger.info("Loaded %d learned predicate(s) from %s.", len(loaded),
                    predicates_file)
        for p in sorted(loaded, key=lambda x: x.name):
            sig = ", ".join(t.name for t in p.types)
            logger.info("  %s(%s)", p.name, sig)

        if os.path.isfile(predicates_file):
            os.makedirs(archive_dir, exist_ok=True)
            self._predicates_cycle_count += 1
            archive_path = os.path.join(
                archive_dir,
                f"cycle_{self._predicates_cycle_count:03d}_predicates.py")
            shutil.copy2(predicates_file, archive_path)
            logger.info("Archived predicates.py to %s.", archive_path)

    # ── Predicate loading ────────────────────────────────────────

    def _load_predicates_from_module_file(
            self, path: str) -> Set[Predicate]:
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

        ctx = build_exec_context(
            types=self._types,
            predicates=self._kept_initial_predicates,
            options=self._get_all_options(),
            extra_context={
                "params": _ParamsView(self),
                "ParamSpec": ParamSpec,
            })

        result, err = exec_code_safely(code, ctx, "LEARNED_PREDICATES")
        if err is not None:
            logger.warning("Failed to load %s:\n%s", path, err)
            return set()
        if not isinstance(result, list):
            logger.warning(
                "%s: LEARNED_PREDICATES must be a list, got %s.", path,
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
    Predicate("WaterBoiled", [jug_type],
              lambda s, objs: s.get(objs[0], "heat_level") >= params["boiled_threshold"]),
]
```

A pre-injected `params` view is in scope; it always reads the **current \
fitted values** of every `ParamSpec` declared in `simulator.py`. Whenever \
MCMC re-fits, predicates picking up `params["name"]` see the new values \
automatically. To share a threshold between a rule and a predicate, declare \
it once in `PARAM_SPECS` and reference `params["name"]` from both.

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

Validate with `evaluate_predicate_quality` (cheap; reports first-flip step, \
monotonicity, coverage on demos). A good milestone predicate flips False→True \
exactly once per goal-reaching demo and stays true. A placement predicate \
should be true exactly when an object is at its intended location and false \
otherwise.

`evaluate_predicate_quality` is also the loader: it updates the predicate \
set used by `evaluate_plan_refinement`. Call it after every edit to \
`predicates.py` before re-running plan refinement.

Predicates persist across online cycles — the file is preserved between \
synthesis sessions. Edit it freely; archives of each cycle's final state \
live in `predicates_archive/`.
"""
