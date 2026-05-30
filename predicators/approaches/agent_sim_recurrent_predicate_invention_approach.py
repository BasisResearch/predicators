"""Recurrent (partial-observability) sim-learning + predicate-invention
approach.

Extends ``AgentSimPredicateInventionApproach`` to handle envs where
some causally-important features are hidden in the agent-visible
observation (motivating example: ``jug.heat_level`` in
``pybullet_boil`` when ``CFG.partially_observable`` is True). The
synthesizing Claude agent now:

* writes rules with a 5-arg signature
  ``rule(state, latent, history, updates, params)`` so they can carry
  a ``latent`` state dict across steps and / or read the prior
  observation history;
* optionally declares ``LATENT_INIT`` (a dict, or zero-arg callable
  returning one) giving the initial latent block;
* invents predicates that may be observation-only OR latent-aware
  (``classifier(state, objs, latent=None)``).

Two natural patterns the prompt presents (agent picks per latent):

* **Counter + threshold** — carry a step counter; flip an observable
  when the counter crosses a learnable threshold.
* **Physical latent + readout** — carry an estimate of the unobserved
  physical quantity; map it through a monotone readout to an
  observable.

A ``State`` flowing through the simulator is one *sample* of the
augmented state — observable features in ``State.data`` plus the
inferred latent dimensions in ``State.latent`` (e.g. ``{"heat": ...}``
or ``{"streak": ...}``); a belief is the (here point-mass) distribution
over such samples.

MCMC fitting threads the latent block across steps within each
trajectory (``compute_sse_recurrent`` / ``fit_params_recurrent``); the
combined simulator used at refinement time threads it through the
opaque ``State.latent`` field so the ``_OracleOptionModel`` interface
``(State, Action) -> State`` stays unchanged *and* backtracking
naturally restores the latent at each search node (``traj[cur_idx]``
carries its own latent; sibling branches don't share it). Latent-aware
predicates work uniformly: ``Predicate.holds`` auto-reads
``state.latent`` when no explicit kwarg is passed, so the same
classifier is correct during ``evaluate_predicate_quality`` *and*
inside ``bilevel_sketch.refine_sketch``.

Example command::

    python predicators/main.py --env pybullet_boil \
        --approach agent_sim_recurrent_predicate_invention --seed 0 \
        --num_train_tasks 10 --num_test_tasks 5 \
        --partially_observable True \
        --num_online_learning_cycles 2 --explorer agent_plan
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pybullet

from predicators.approaches.agent_sim_predicate_invention_approach import \
    AgentSimPredicateInventionApproach
from predicators.code_sim_learning.training import ParamSpec, \
    compute_sse_recurrent, fit_params_recurrent
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules_with_latent, init_latent, merge_updates, read_latent_init
from predicators.structs import Action, LowLevelTrajectory, State, Task

logger = logging.getLogger(__name__)


class AgentSimRecurrentPredicateInventionApproach(
        AgentSimPredicateInventionApproach):
    """Partial-observability variant: rules carry a `latent` block across
    steps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Loaded from simulator.py's LATENT_INIT export (None ⇒ no
        # latent state; the block stays an empty dict).
        self._latent_init: Any = None
        # Cached during `_learn_simulator` so `_fit_parameters` can
        # regroup the flat `base_pred_triples` back into per-trajectory
        # chunks (latent threads within a trajectory, not across).
        self._fit_trajectories: List[LowLevelTrajectory] = []

    @classmethod
    def get_name(cls) -> str:
        return "agent_sim_recurrent_predicate_invention"

    # ── Synthesis-loading overrides ──────────────────────────────

    def _learn_simulator(self, trajectories: List[LowLevelTrajectory]) -> None:
        """Same flow as the parent, with a trajectory cache for fitting."""
        self._fit_trajectories = list(trajectories)
        try:
            super()._learn_simulator(trajectories)
        finally:
            self._fit_trajectories = []

    def _load_simulator_from_module_file(  # type: ignore[override]
        self,
        path: str,
        trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Tuple[Optional[List], Optional[List[ParamSpec]], Optional[Dict[
            str, List[str]]], Optional[Dict[str, Any]]]:
        """Load rules/specs/features plus LATENT_INIT from one exec.

        Reads ``LATENT_INIT`` from the exec namespace the parent now
        returns (its 4th element) rather than re-execing the file.
        Overrides the parent's ``@staticmethod`` form because we need
        ``self`` to stash ``LATENT_INIT`` on the instance (the ``# type:
        ignore[override]`` is for the static-vs-instance mismatch; Python
        dispatches through ``self.`` correctly).
        """
        result = super()._load_simulator_from_module_file(path, trajectories)
        rules, specs, features, ns = result
        self._latent_init = None
        if isinstance(ns, dict):
            self._latent_init = read_latent_init(ns)
        if self._latent_init is not None:
            n_keys = (len(self._latent_init) if isinstance(
                self._latent_init, dict) else 0)
            logger.info("Loaded LATENT_INIT with %d key(s) from %s.", n_keys,
                        path)
        return rules, specs, features, ns

    # ── Parameter fitting (recurrent SSE) ────────────────────────

    def _fit_parameters(  # type: ignore[override]
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        process_features: Dict[str, List[str]],
    ) -> Tuple[Dict[str, float], float]:
        """MCMC over the recurrent (per-trajectory) SSE.

        Re-groups the flat ``base_pred_triples`` back into per-
        trajectory chunks using lengths cached in
        ``self._fit_trajectories``. If no trajectory info is available
        (e.g. the recurrent approach is invoked through a call site that
        didn't go through ``_learn_simulator``), falls back to treating
        the whole input as one trajectory — the latent then threads
        across the entire history, which is wrong but unlikely to crash.
        """
        groups = self._group_triples_by_trajectory(base_pred_triples)
        if not groups:
            logger.warning("No trajectory groups for recurrent fitting; "
                           "falling back to single-trajectory rollout.")
            groups = [base_pred_triples]

        latent_init = self._latent_init
        init_params = {s.name: s.init_value for s in specs}
        pre_sse = compute_sse_recurrent(rules, groups, init_params,
                                        latent_init, process_features)
        logger.info("Recurrent fit — pre-SSE: %.6f", pre_sse)

        result = fit_params_recurrent(
            rules=rules,
            trajectories=groups,
            param_specs=specs,
            latent_init=latent_init,
            process_features=process_features,
        )
        fitted_params = result.point_estimate
        post_sse = compute_sse_recurrent(rules, groups, fitted_params,
                                         latent_init, process_features)
        logger.info("Recurrent fit — post-SSE: %.6f", post_sse)
        for name in sorted(fitted_params):
            init_val = init_params[name]
            fit_val = fitted_params[name]
            delta = fit_val - init_val
            pct = (delta / init_val * 100) if init_val != 0 else float("nan")
            logger.info("  %-30s  %.4f -> %.4f  (Δ=%.4f, %+.1f%%)", name,
                        init_val, fit_val, delta, pct)
        return fitted_params, post_sse

    def _group_triples_by_trajectory(
        self,
        triples: List[Tuple[State, Action, State]],
    ) -> List[List[Tuple[State, Action, State]]]:
        """Slice the flat triples list back into per-trajectory groups."""
        if not self._fit_trajectories:
            return []
        lengths = [len(t.actions) for t in self._fit_trajectories]
        if sum(lengths) != len(triples):
            logger.warning(
                "Trajectory-length mismatch (sum=%d vs triples=%d); "
                "skipping grouping.", sum(lengths), len(triples))
            return []
        groups: List[List[Tuple[State, Action, State]]] = []
        idx = 0
        for n in lengths:
            groups.append(triples[idx:idx + n])
            idx += n
        return groups

    # ── Combined simulator (latent-threaded via state.latent) ────

    def _build_combined_simulator(  # type: ignore[override]
        self,
        learned_simulator: LearnedSimulator,
    ) -> Callable[[State, Action], State]:
        """Compose base env + recurrent rules; carry latent on state.latent.

        The latent block is part of the planning state via the opaque
        ``State.latent`` field (see ``structs.State``), so backtracking
        naturally restores it at each search node — ``traj[cur_idx]``
        carries its own latent. The simulator reads ``state.latent`` on
        entry, threads it through the agent's rules, and attaches the
        updated latent to the returned state. No closure cache, no
        hash-key collisions on observationally-identical states.

        First-step robustness: if ``state.latent`` is None (e.g. the
        initial state wasn't attached by :meth:`_attach_initial_latent`,
        or a caller passed a fresh State), fall back to LATENT_INIT.
        """
        del learned_simulator  # we use the raw rules + params directly
        assert self._process_rules is not None, (
            "_build_combined_simulator called before PROCESS_RULES loaded")
        rules: List = self._process_rules
        latent_init = self._latent_init
        # Hold a reference to the dict, not its current values, so MCMC
        # updates to params are picked up by the closure live.
        params = self._fitted_params

        def combined_simulate(state: State, action: Action) -> State:
            # `state` is one sample of the augmented state: observable
            # features in `.data` + inferred latent dims in `.latent`.
            # This is a per-sample transition (obs, latent) ->
            # (obs', latent'); the belief is the (here point-mass)
            # distribution over such samples. Copy the incoming latent
            # so sibling branches at the same parent don't share a dict.
            latent = (dict(state.latent) if state.latent is not None else
                      init_latent(latent_init, params))
            try:
                base_state = self._base_env.simulate(state, action)
            except pybullet.error as e:
                logger.warning(
                    "PyBullet error in recurrent combined_simulate (%s); "
                    "recreating base env and retrying.", e)
                self._recreate_base_env()
                base_state = self._base_env.simulate(state, action)
            # History: single-step window. The closure has no access
            # to the planner's full action sequence, so rules that
            # need long history must encode it in ``latent``
            # incrementally (the standard recurrent pattern).
            history: List[Tuple[State,
                                Optional[Action]]] = [(base_state, action)]
            updates = apply_rules_with_latent(base_state, latent, history,
                                              rules, params)
            next_state = (merge_updates(base_state, updates)
                          if updates else base_state)
            next_state.latent = latent
            return next_state

        return combined_simulate

    # ── Initial-latent seeding for refinement ────────────────────

    def _attach_initial_latent(self,
                               task: Task) -> Task:  # type: ignore[override]
        """Seed ``task.init.latent`` with the initial latent block.

        Refinement starts here: the planner's ``traj[0]`` is
        ``task.init``, and we want ``combined_simulate`` to find a well-
        formed latent on it. If the agent's ``simulator.py`` did not
        export a ``LATENT_INIT`` (or the resulting block is empty
        because every value was a no-op), we leave the task alone so
        downstream code keeps the legacy ``state.latent is None``
        behaviour.
        """
        if self._latent_init is None:
            return task
        initial_latent = init_latent(self._latent_init, self._fitted_params
                                     or {})
        if not initial_latent:
            return task
        init_state = task.init.copy()
        init_state.latent = initial_latent
        return Task(init=init_state,
                    goal=task.goal,
                    alt_goal=task.alt_goal,
                    goal_nl=task.goal_nl)

    # ── Latent materialisation for predicate evaluation ──────────

    def materialise_latent(
        self,
        traj: LowLevelTrajectory,
    ) -> List[Optional[Dict[str, Any]]]:
        """Roll a trajectory through the agent's rules; return per-step latent.

        Used by :func:`evaluate_predicate_quality` so latent-aware
        predicates can be scored against meaningful latent values.

        Returned list aligns with ``traj.states`` (len = num states).
        Entry ``i`` is the latent *before* evaluating predicates at
        state ``i`` — i.e. after rolling the simulator through the
        first ``i`` actions. Entry 0 is the freshly-initialised
        latent. If no rules are loaded yet, every entry is ``None`` so
        latent-aware classifiers fall back to their default branch.
        """
        if not self._process_rules:
            return [None] * len(traj.states)
        rules = self._process_rules
        params = self._fitted_params
        latent = init_latent(self._latent_init, params)
        out: List[Optional[Dict[str, Any]]] = [dict(latent)]
        history: List[Tuple[State, Optional[Action]]] = []
        for i in range(len(traj.actions)):
            state = traj.states[i]
            action = traj.actions[i]
            history.append((state, action))
            try:
                apply_rules_with_latent(state, latent, history, rules, params)
            except Exception:  # pylint: disable=broad-except
                # If a rule crashes, fall back to None for the
                # remaining steps so predicate evaluation continues.
                out.extend([None] * (len(traj.states) - len(out)))
                return out
            out.append(dict(latent))
        return out

    # ── Prompt overrides ─────────────────────────────────────────

    def _extra_synthesis_system_prompt(self) -> str:
        base = super()._extra_synthesis_system_prompt()
        return base + "\n\n" + _RECURRENT_PROMPT_SECTION

    def _extra_synthesis_message(self, extra_paths: Dict[str, str]) -> str:
        base = super()._extra_synthesis_message(extra_paths)
        return base + "\n\n" + _RECURRENT_MESSAGE_SECTION


_RECURRENT_PROMPT_SECTION = """\
## Recurrent rules (partial observability)

This approach handles partial observability: the observation may omit
causally-important quantities — there may be several, one, or none.
Anything omitted is *absent entirely* from the state (it appears under
no name, not even as a NaN placeholder), so you cannot read it and
must *infer* its existence and dynamics from how the observable
features evolve. Inspect the trajectories first to judge how many
latents (if any) you need: a feature that drifts or ramps with no
visible observed driver is likely downstream of an accumulating
latent; if every observable is already explained by other observed
quantities, you need no latent at all (write ordinary 3-arg rules).
One common case: a hidden continuous quantity surfaced only through a
derived observable that ramps once the latent crosses a threshold.

Model the hidden state explicitly: each ``State`` you predict is one
sample of an *augmented* state — observable features in ``state.data``
plus the latent dimensions you infer in ``state.latent`` (a free-form
dict like ``{"level": 0.73}`` or ``{"count": 22}``). Write rules with
the recurrent 5-arg signature so they can read and advance that latent:

```python
def my_rule(state, latent, history, updates, params):
    # state    : current observation State (no hidden features)
    # latent   : Dict[str, Any], mutated in place — the latent state
    #            dims you track, threaded across steps
    # history  : List[Tuple[State, Optional[Action]]], read-only;
    #            most recent last; first action is None
    # updates  : ProcessUpdate dict, also mutated in place
    # params   : Dict[str, float] (fitted scalars)
    ...
    return updates
```

The 2nd parameter MUST be named ``latent`` — the engine inspects each
rule's signature and only threads the latent block into rules that
declare it. Declare the initial latent block:

```python
LATENT_INIT = {"level": 0.0, "count": 0}
# OR a zero-arg callable returning such a dict.
# Use ParamSpec("name", ...) values to make an init value learnable.
```

Legacy 3-arg `rule(state, updates, params)` rules still work — the
engine inspects each rule's signature. Mix both styles freely.

The type, feature, latent, and parameter names in the examples below
(`widget`, `fixture`, `progress`, `level`, ...) are illustrative — use
whatever the inspect tools actually report for your task.

### Two synthesis patterns (agent picks per latent)

**Pattern A — Counter + threshold.** Carry a step counter; flip the
observable when it crosses a learnable threshold. Same statistical
shape as a delayed discrete event:

```python
PARAM_SPECS = [ParamSpec("delay", init_value=33, lo=1, hi=200)]
LATENT_INIT = {"count": 0}

def count_rule(state, latent, history, updates, params):
    active = is_widget_at_fixture(state)  # observable check
    fixture_on = state.get(fixture, "is_on") > 0.5
    if active and fixture_on:
        latent["count"] += 1
    else:
        latent["count"] = 0
    fired = latent["count"] >= params["delay"]
    updates[widget]["progress"] = 1.0 if fired else 0.0
    return updates
```

**Pattern B — Physical latent + readout.** Carry an estimate of the
unobserved quantity; map it through a (typically monotone) function to
predict the observable. Higher resolution: the observable co-varies
smoothly with the latent before the symbolic "done" point.

```python
PARAM_SPECS = [ParamSpec("rate", init_value=0.03, lo=0.0, hi=0.1)]
LATENT_INIT = {"level": 0.0}

def level_rule(state, latent, history, updates, params):
    active = is_widget_at_fixture(state)
    fixture_on = state.get(fixture, "is_on") > 0.5
    if active and fixture_on:
        latent["level"] += params["rate"]
    lvl = latent["level"]
    # monotone readout: ramps from 0 once `lvl` passes an onset (~0.85)
    updates[widget]["progress"] = max(0.0, min(1.0, (lvl - 0.85) / 0.15))
    return updates
```

**How to choose.** Look at the derived observable in the inspect
tools:
- Smooth ramp across many steps ⇒ Pattern B (partial-progress
  signal; rate identifiable from a single trajectory by slope-fit).
- Clean discrete flip at a variable tick ⇒ Pattern A may suffice
  (one learnable threshold, calibrated from the empirical
  flip-time distribution across trajectories).
- Mixing is fine: different rules / different latents can use
  different patterns within the same simulator.

### Predicate signature

Classifiers may stay observation-only or take an optional ``latent``
kwarg. The latent block is available at refinement time too — the
planner threads it through ``state.latent`` across search nodes, and
``Predicate.holds`` auto-routes it into classifiers that opted in. Be
defensive: at the very first step ``state.latent`` may still be ``{}``
if the agent's ``LATENT_INIT`` is empty, and during predicate-quality
scoring on *raw env* trajectories ``latent`` will be the block
materialised by the agent's rules (so still meaningful, but only as
accurate as the rules themselves).

```python
# Observation-only (robust to bad rule chains; preferred when the
# observable carries enough signal):
Predicate("ProcessDone", [widget_type],
          lambda s, objs, latent=None:
              s.get(objs[0], "progress") > 0.5)

# Latent-aware (inherits simulator correctness; defend against
# missing keys at step 0):
Predicate("ProcessDone", [widget_type],
          lambda s, objs, latent=None:
              (latent or {}).get("level", 0.0) >= params["done_thresh"])
```

The kwarg MUST be named exactly ``latent`` for the auto-routing to
fire. Trade-off: latent-aware predicates inherit the simulator's
correctness; observation-only predicates are robust to bad rules
but only work when the observable carries enough signal.

### Diagnostics

`evaluate_predicate_quality` rolls each trajectory through your
simulator to materialise the latent before scoring classifiers, so
latent-aware predicates get a real block there. Use the eval
report to localise failures (bad rule chain vs. bad threshold).
"""

_RECURRENT_MESSAGE_SECTION = """\
## Partial observability

Some causally-important quantities may be absent from the agent-visible
observation entirely (under no name, not even as NaN) — possibly
several, possibly none. Inspect the trajectories first to judge whether
any hidden process is at work and which observable features are your
window into it; then, if any latents are needed, choose Pattern A or
Pattern B (or mix) to model the underlying dynamics in `latent`.
"""
