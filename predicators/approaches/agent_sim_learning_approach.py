"""Agent sim-learning approach: learns a simulator program online.

Extends AgentModelBasedApproach to learn process dynamics via an
agent-synthesized step-level simulator with parameterized process
rules. Parameters are fitted via emcee ensemble MCMC (training.py).

The approach creates a base oracle (PyBullet with process
dynamics disabled) and composes it with the learned step-level
dynamics into a single simulator function, plugged into a standard
_OracleOptionModel for true per-step interleaving.

Example command::

    python predicators/main.py --env pybullet_boil \
        --approach agent_sim_learning --seed 0 \
        --num_train_tasks 10 --num_test_tasks 5 \
        --num_online_learning_cycles 5 --explorer agent_plan
"""

import copy
import dataclasses
import hashlib
import inspect
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Collection, Dict, FrozenSet, Iterator, \
    List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.tools import SAMPLER_SYNTHESIS_TOOL_NAMES, \
    SYNTHESIS_TOOL_NAMES, _SnapshotTarget, create_synthesis_tools, \
    evaluate_states_with, finalize_versioned_snapshot, \
    make_write_snapshot_hook
from predicators.agent_sdk.tools.inspection import render_options_digest, \
    render_trajectory_digest, render_types_digest
from predicators.approaches.agent_model_based_approach import \
    AgentModelBasedApproach
from predicators.approaches.sampler_learning_mixin import SamplerLearningMixin
from predicators.approaches.synthesis_validation import \
    build_candidate_option_model
from predicators.code_sim_learning.active_experiment import laplace_ensemble, \
    mean_bernoulli_entropy, perturbation_ensemble, \
    posterior_subsample_ensemble
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec
from predicators.code_sim_learning.fitting import FIT_NOISE_SIGMA, \
    compute_sse, compute_sse_recurrent, fit_rule_parameters, \
    fit_rule_parameters_latent, log_param_changes, log_sse_breakdown
from predicators.code_sim_learning.identifiability import \
    format_identifiability, identifiability_report, \
    select_trustworthy_params
from predicators.code_sim_learning.physical_sysid import fit_params_rollout, \
    fit_params_rollout_trimmed
from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    dispose_env, physical_param_anchors
from predicators.code_sim_learning.rollout_objective import compute_rollout_sse
from predicators.code_sim_learning.trajectory_prep import \
    compute_residual_scaling, split_at_rest_points, truncate_settled_tail
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, apply_rules_with_latent, has_latent_rules, init_latent, \
    iter_feature_residuals, merge_updates, read_latent_init, \
    read_physical_param_specs, read_simulator_components, \
    stamp_physical_spec_scales
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_simulator
from predicators.option_model import _OptionModelBase, _OracleOptionModel
from predicators.settings import CFG
from predicators.structs import Action, Dataset, DerivedPredicate, \
    GroundAtom, InteractionResult, LowLevelTrajectory, ParameterizedOption, \
    Predicate, State, Task, Type, step_option_labels

logger = logging.getLogger(__name__)

# Canonical "### Rule signature" block for the synthesis system prompt
# (fully-observable / legacy 3-arg). Spliced in at the
# ``__RULE_SIGNATURE_SECTION__`` placeholder by
# ``_build_synthesis_system_prompt``; the partial-observability subclass
# overrides ``_rule_signature_section`` to swap in the recurrent 5-arg
# form so its prompt never shows the 3-arg signature as canonical.
_FO_RULE_SIGNATURE_SECTION = '''\
### Rule signature

```python
def rule(state, updates, params):
    # state:   the current env State
    # updates: Dict[Object, Dict[str, float]] accumulated from prior rules
    # params:  Dict[str, float], one entry per ParamSpec
    #
    # Accumulate, don't replace:
    #     updates.setdefault(obj, {})[feat] = new_value
    # Return the same dict.
    ...
```'''

# Synthesis system prompt, rendered by
# ``AgentSimLearningApproach._build_synthesis_system_prompt``: the
# ``__UPPER_SNAKE__`` placeholders are substituted per instance
# (observability, env parameter menu, tool surface, subclass extras).
_SYNTHESIS_SYSTEM_PROMPT_TEMPLATE = """\
You are synthesizing a parameterized process-dynamics simulator for a \
robotic manipulation environment.

A separate PyBullet base sim handles robot movement, grasping, and rigid- \
body physics. Your simulator handles **process dynamics** - features \
that change due to physical or causal processes (gradual level changes, \
accumulation, propagation between contacting objects, sensor readouts \
that lag actuators, etc.) that the base sim doesn't model.

## What you produce

One file `simulator.py` (path given in the first message) defining three \
top-level names:

```python
PROCESS_RULES:    List[Callable]            # rule functions (see signature below)
PARAM_SPECS:      List[ParamSpec]           # learnable parameters
PROCESS_FEATURES: Dict[str, List[str]]      # {type_name: [feature_names]} your rules predict
```

`PROCESS_FEATURES` defines both the loss scope and the test-time overwrite \
scope: only the listed `(type, feature)` pairs are scored against \
observations, and only those are written on top of the base sim at test \
time. Be honest - listing features your rules don't actually update \
inflates the loss without giving MCMC anything to optimise.
__PHYSICAL_PARAMS_SECTION__
__RULE_SIGNATURE_SECTION__

### Multiple objects of the same type

A task may contain **several objects of the same type** - two widgets, \
three fixtures, or one of each - and the count varies from task to task. \
Your rules run once per step over the entire `State`, so they must act on \
*whatever objects are present*, never a hard-coded slot. Code like \
`widgets[0]` silently ignores every other instance and breaks the moment \
a task has more (or fewer) objects than the trajectory you calibrated on.

Gather the relevant objects by type and loop over the binding(s) the rule \
acts on, emitting updates keyed by the specific object the effect applies \
to:

```python
widgets  = [o for o in state.data if o.type.name == "widget"]
fixtures = [o for o in state.data if o.type.name == "fixture"]
for widget in widgets:
    for fixture in fixtures:           # all pairs, or pair each widget
        if at_fixture(state, widget, fixture, params):   # to its nearest
            wv = state.get(widget, "progress")
            updates.setdefault(widget, {})["progress"] = wv + params["rate"]
```

The same `params` apply to every object of a type: you are learning the \
shared physics of "a widget", not per-instance constants. If a rule \
genuinely needs exactly one object (a single global clock, say), assert \
that rather than silently indexing `[0]`.

### Timing

Each rule fires once per step:

```
state[t] ──base_sim──▶ draft state[t+1] ──your rules──▶ final state[t+1]
                                               ^^^^^^^
                        (only PROCESS_FEATURES are overwritten)
```

Rules see `state[t]`. They cannot see actions, the base sim's draft, or \
`state[t+2]`. If a feature changes one step *after* its gating event \
(e.g. an action toggles a gating flag at `t`, but the feature it drives \
only starts changing at `t+1`), that's an inherent 1-step lag in the \
data - accept the single boundary residual or model the delay with an \
extra parameter rather than chasing it with ever-stricter conditions.

### Geometric gates

If a rule's firing condition depends on the relative position of two \
bodies, do **not** gate on the raw distance between their recorded \
poses. `obj.x, obj.y` is the recorded pose origin - usually a body's \
base or frame center - while the point that actually drives the \
physics (a contact surface, an outlet on the body's side, an \
end-effector tip, a container opening, a handle) is typically offset \
from it. That offset lives in the body's **local frame**, so it \
rotates with the body's `rot` feature; gating on raw origin distance \
silently bakes in one task's orientation and breaks on any task where \
the fixture is rotated differently.

**Default to a learned, rotation-aware anchor offset.** Express every \
two-body geometric gate as a distance to an *anchored* point - the \
fixture origin plus a local-frame offset rotated into the world frame \
by the fixture's `rot` - with the offset declared as learnable params:

```python
PARAM_SPECS = [
    # Functional point offset, in the fixture's LOCAL frame:
    ParamSpec("fixture_local_dx",       0.0,  lo=-0.3, hi=0.3),
    ParamSpec("fixture_local_dy",       0.0,  lo=-0.3, hi=0.3),
    ParamSpec("widget_at_fixture_dist", 0.10, lo=0.0,  hi=0.4),
]

# `fixture`, `widget`: the relevant object pair (bind as your rule needs).
__PROCESS_RULE_SIGNATURE__
    rot = state.get(fixture, "rot")
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    local_offset = np.array([params["fixture_local_dx"],
                             params["fixture_local_dy"]])
    origin = np.array([state.get(fixture, "x"), state.get(fixture, "y")])
    anchor = origin + rot_mat @ local_offset  # world-frame point
    widget_xy = np.array([state.get(widget, "x"), state.get(widget, "y")])
    if np.linalg.norm(widget_xy - anchor) < params["widget_at_fixture_dist"]:
        ...  # fire
```

If the functional point really does coincide with the recorded origin, \
the fit drives the offsets to ~0 - no harm done. A threshold-only gate \
(no offset) is the exception: use one only after you have positively \
confirmed the recorded origin *is* the functional point. Share the \
offset and distance params with the gating predicate so the rule and \
predicate anchor to the same point.

**Required check before committing a geometric gate.** Bucket the \
trajectory steps by whether the gated effect actually fired, compute \
your gate quantity at each step, and confirm the two buckets separate \
by a clear margin. If they overlap, or separate only by a knife-edge \
gap (~5% of the value range or narrower), the gate references the \
wrong point - a threshold flush against the data boundary is a \
rejected fit, not a fit. Do **not** nudge the threshold to paper over \
it: add or refit the anchor offset and re-bucket. To find the offset, \
__SCENE_VIZ_HINT__; the gap \
between the origin and the effect-firing cluster is the offset.

### ParamSpec

```python
ParamSpec(name: str, init_value: float,
          lo: Optional[float] = None, hi: Optional[float] = None)
```

Bounds shape both the MCMC prior and the warm-start clamp. Set `lo=0.0` \
for non-negative rates, etc.

### Pre-injected when `simulator.py` is exec'd

`numpy as np`, `ParamSpec`. Import anything else at the top of the file. \
The data classes (`State`, `Object`, `Action`, ...) come from \
`predicators.structs`; source is in the reference file linked in the \
first message.

## Tools

`Write` / `Edit` `simulator.py` is your normal coding loop. Every \
successful write is snapshotted to \
`simulator_versions/cycle_XXX_vers_YYY_simulator.py` (deduped by \
content; ``XXX`` is the current cycle, ``YYY`` resets per cycle). \
`sim.fit` / `sim.residuals` (and the probe's candidate-model refit) \
load the file fresh on every call and prefix their reports with \
`[cycle_XXX_vers_YYY]` so you and reviewers can diff iterations.

- `run_python(code)` - ad-hoc data exploration AND validation. \
`trajectories`, `np`, `ParamSpec` in scope; when the learn message \
states a task objective, `evaluate_trajectory(states, actions=None, \
task_idx=0)` scores a state sequence with the env's ground-truth \
evaluator (returns reward / solved; on your own simulator's rollouts \
the verdict is only as good as the simulator). The `sim` probe over \
your CANDIDATE simulator also lives here: `sim.fit()` (parameter \
fitting + report; cheap inner-loop signal), `sim.residuals()` \
(per-feature breakdown: mismatch counts, mean / max abs error, \
vs-baseline improvement (negative ⇒ rules are adding error), worst-N \
example transitions - diagnostic for *which* rule to fix), \
`sim.refine` (backtracking parameter search on a plan sketch), \
`sim.run` (forward rollout with subgoal checking). **Does not** \
define rules.

`sim.fit` and the refine-then-run protocol test complementary \
things - pointwise accuracy vs. goal reachability. A rule can have \
ε-small SSE and still get a saturation threshold or alignment cap *just* \
wrong enough that refinement can't satisfy a subgoal. Use `sim.fit` + \
`sim.residuals` as the fast inner loop, and refine-then-run as the \
slow goal-relevant gate before declaring done.

### Refinement vs. forward validation (read before tuning a threshold)

Validation is two checks under the same option model. `sim.refine` \
samples continuous params with up to 50 attempts per \
parametric step and snapshots state at each backtrack - failures are \
isolated per step. The forward pass - one continuous `sim.run` of the \
refined plan, state carrying forward across all options, subgoal \
annotations checked per step - matches how test time will execute it. \
Any divergence between the \
two indicates the learned model is *more permissive* than the env's \
effective behavior: refinement's looser gates accept a Place/Wait \
that the env-driven rollout won't actually achieve.

When `sim.refine` passes but the continuous `sim.run` reports a \
`SUBGOAL NOT REACHED` (or the goal check fails), the failure mode is \
almost always one of these:

1. **A learned gate threshold is wider than the env's effective \
threshold.** Example: the env's process rule only fires when the \
widget-to-fixture distance < 0.05, but you set \
`widget_at_fixture_dist = 0.063` for "safety margin". Refinement \
accepts a Place at distance 0.05–0.063 (your `WidgetAtFixture` \
predicate is true and your learned rule fires); forward validation \
runs the same Place, the env's rule never fires (distance > env \
threshold), and Wait runs to its step cap without `WidgetReady` \
holding. **Fix:** tighten the gate to match the env's empirical \
boundary, do not widen for slack.
2. **A wait-termination cutoff fires before the env-side feature \
catches up.** Example: `WidgetReady = process_value >= 0.99` fires at \
the learned simulator's step 34 (process_value=0.9996), but the env's \
goal-check requires the underlying feature to reach 1.0 - refinement's \
subgoal passes, but the final-state goal check on env state fails. \
**Fix:** align the predicate's cutoff with the env's effective \
cutoff, *and* confirm by re-running plan refinement after the change.

**Rule of thumb:** when in doubt, *tighten* learned thresholds toward \
the env's empirical boundary, never loosen them. Widening hides \
discrepancies during refinement and reveals them at test time as \
0-solve regressions.
__SYNTHESIS_PROMPT_EXTRA__
## Plan format for `sim.refine` / `sim.run`

One option call per line, **with every option argument supplied and using \
typed object references** (`obj:type`), matching exactly the Options \
digest in your prompt. Use that digest (or `run_python` over a trajectory) \
to read off the right names and arities - the parser is strict and \
silently omitting an argument will not be auto-filled. Example:

```
PickWidget(robot:robot, widget0:widget)
Place(robot:robot) -> {WidgetAtFixture(widget0:widget, fixture0:fixture)}
ActivateFixture(robot:robot, fixture0:fixture)
Wait(robot:robot) -> {WidgetReady(widget0:widget)}
...
```

(The names above are illustrative - use whatever options, types, and \
predicates your prompt digests actually list for your task.) Insert a \
`Wait` after any action that triggers a delayed process (gradual \
accumulation, propagation, sensor catch-up) so your rules have steps to \
fire on.

**Subgoal annotations** (`-> {Atom(obj:type, ...)}` after a step) are \
optional in general but **effectively required after open-ended skills \
like `Place`**. Without one the backtracking search has no preference for \
*where* to put the object, so a `Place; Wait` pair will refine cleanly \
but skip past the relevant target location and your rules never fire - \
the run looks like a rule bug but is actually a missing subgoal. For \
`Wait`, the annotation also specifies when the wait should terminate; \
prefix an atom with `NOT` if it should become false.

## Workflow

1. Explore data with `run_python` - what features change per step, \
which ones aren't explained by the base sim.
2. `Write` `simulator.py`; `Edit` to iterate.
3. Score with `sim.fit()`, then `sim.residuals()` to find \
diverging features. Negative `vs base` ⇒ a rule is actively hurting - \
usually a wrong gate or sign.
4. When SSE is plausible, propose an option-skeleton plan and validate: \
`sim.reset(task_idx=i)`, `sim.refine(plan, require_goal=True)`, then a \
continuous `sim.run` of the refined plan from a fresh \
`sim.reset(task_idx=i)`. A stuck refine step means the rules gating \
its subgoal atoms are too tight or too loose; a refine-pass whose \
`sim.run` diverges means a rule is too permissive. Fix and \
re-validate - do not declare done until BOTH pass.
"""


@dataclasses.dataclass(frozen=True)
class _SynthesisPaths:
    """Host- and agent-visible paths for one synthesis session.

    ``simulator_file`` / ``versions_dir`` are host paths the harness
    reads and writes; ``simulator_file_for_agent`` /
    ``sandbox_dir_for_agent`` are how the sandboxed agent must refer to
    the same locations (see ``_resolve_synthesis_paths``).
    """
    base: str
    simulator_file: str
    versions_dir: str
    simulator_file_for_agent: str
    sandbox_dir_for_agent: Optional[str]


# ── Approach ─────────────────────────────────────────────────────


class AgentSimLearningApproach(SamplerLearningMixin, AgentModelBasedApproach):
    """Bilevel planning with a learned step-level simulator.

    During online learning:
    1. Collect trajectories (inherited from AgentModelBasedApproach)
    2. Segment into option-level transitions
    3. Synthesize parameterized process rules via Claude agent
    4. Fit rule parameters via emcee ensemble MCMC
    5. Compose with base oracle into a combined simulator
    6. Build _OracleOptionModel with the combined simulator

    During solving:
    - Uses the learned model for plan validation in backtracking
      refinement.

    Per-skill sampler learning (mode resolution, synthesis session
    plumbing, loading) lives in :class:`SamplerLearningMixin`.
    """

    # Allowlist of env predicate names surfaced to the agent; None keeps
    # every env predicate. CFG.agent_sim_learn_kept_predicates_names
    # overrides this class default when non-empty, so an experiment can
    # strip predicates - even goal predicates - from the agent's
    # vocabulary (prompts, tools, subgoal annotations) without touching
    # env-side goal checking or the task evaluator. Tasks whose goal
    # atoms are stripped must carry ``goal_nl``: the natural-language
    # goal becomes the agent's only goal signal.
    KEPT_INITIAL_PREDICATE_NAMES: Optional[FrozenSet[str]] = None

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 *args: Any,
                 option_model: Optional[_OptionModelBase] = None,
                 **kwargs: Any) -> None:
        # Pass the option model in so the parent __init__ doesn't spin up
        # its own full-process env, which would fight this one for the
        # PyBullet GUI client.
        self._base_env = create_new_env(CFG.env,
                                        do_cache=False,
                                        use_gui=CFG.option_model_use_gui,
                                        skip_process_dynamics=True)
        if option_model is None:
            option_model = _OracleOptionModel(initial_options,
                                              self._base_env.simulate)
            option_model.sim_env = self._base_env
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         *args,
                         option_model=option_model,
                         **kwargs)
        # Capture-validation rollouts each run on a freshly constructed env
        # (see ToolContext.validation_env_scope): repeats on the shared
        # ``_base_env`` are correlated across resets, so only fresh envs
        # sample the distribution the real episode will.
        self._tool_context.validation_env_scope = \
            self._fresh_validation_env_scope
        # Env predicates surfaced to the agent (see
        # KEPT_INITIAL_PREDICATE_NAMES). Computed once here; everything
        # agent-facing flows through _get_all_predicates().
        self._kept_initial_predicates: Set[Predicate] = (
            self._compute_kept_initial_predicates())
        if self._resolve_kept_names() is not None:
            kept_names = sorted(p.name for p in self._kept_initial_predicates)
            stripped = sorted(p.name for p in self._initial_predicates
                              if p not in self._kept_initial_predicates)
            logger.info(
                "Predicate stripping: kept %s; stripped (hidden from the "
                "agent): %s", kept_names, stripped)
            missing_nl = [
                i for i, t in enumerate(self._train_tasks)
                if not t.goal_nl and any(
                    a.predicate not in self._kept_initial_predicates
                    for a in t.goal)
            ]
            assert not missing_nl, (
                f"Stripping hides goal predicates from the agent, so the "
                f"affected tasks must supply `goal_nl` as the goal signal. "
                f"Missing on train task indices: {missing_nl}")
        self._learned_simulator: Optional[LearnedSimulator] = None
        # Loss-scope mask for parameter fitting (compute_sse).
        self._process_features: Dict[str, List[str]] = {}
        self._process_rules: Optional[List] = None
        # Always the same dict object: fits update it in place via
        # clear()+update() so _ParamsView (held by invented predicate
        # classifiers) picks up new values without holding a reference to
        # ``self``. Truthy iff a fit has populated it.
        self._fitted_params: Dict[str, float] = {}
        # ParamSpecs of the most recently fitted simulator (names + bounds);
        # kept so the active-experiment ensemble can perturb each param
        # within its declared box. Parallel to ``_fitted_params``.
        self._param_specs: List[ParamSpec] = []
        # Small ensemble of plausible parameter vectors, rebuilt after
        # every fit when active-experiment exploration is on. When a
        # posterior fit exists, member 0 is that fit's MAP; otherwise it
        # falls back to ``_fitted_params``. Empty when info-seeking is
        # disabled or no fit has run yet.
        self._param_ensemble: List[Dict[str, float]] = []
        # Full result used for ensemble calibration. Usually this is the
        # solver fit; when info-seeking runs extra MCMC, it is the
        # exploration-only posterior. ``None`` after an oracle-param run.
        self._last_fit_result: Optional[FitResult] = None
        self._fit_sse: float = float("inf")
        self._learning_mode: bool = False
        # Snapshot tags of the most recent simulator / predicates files
        # committed by the synthesis agent, used to stamp newly collected
        # online trajectories with their source-version provenance
        # (consumed in the next learn-phase prompt).
        self._current_simulator_version: Optional[str] = None
        self._current_predicates_version: Optional[str] = None
        self._init_sampler_learning_state()
        # Partial-observability latent block: loaded from a simulator's
        # LATENT_INIT export (None ⇒ no latent state). When the loaded
        # rules use the recurrent 5-arg signature, fitting, the combined
        # simulator, and the SSE diagnostics thread this latent across
        # steps; legacy 3-arg rules ignore it entirely (fully-observable
        # behavior is unchanged). Dispatch keys off the rule signatures
        # via ``has_latent_rules``, not this field.
        self._latent_init: Any = None
        # Cached per learn cycle so recurrent fitting can regroup the flat
        # base_pred_triples back into per-trajectory chunks (latent
        # threads within a trajectory, not across).
        self._fit_trajectories: List[LowLevelTrajectory] = []
        # System identification: PHYSICAL_PARAMS export (agent-declared
        # sparse subset of self._base_env.get_physical_param_info()),
        # identified values applied in place to the base env. The rollout
        # fit itself builds a fresh headless env per rollout (see
        # _get_rollout_fit_env), never touching the planning base env.
        self._physical_param_specs: List[ParamSpec] = []
        self._identified_physical_params: Dict[str, float] = {}
        # Explainability (trimming) verdicts are memoized per learn phase
        # (cleared when _fit_trajectories is refreshed): repeated
        # sim.fit calls with the same declaration signature
        # reuse the sweep instead of re-rolling it, which both saves
        # rollouts and pins the verdict for identical inputs.
        self._explainability_cache: Dict[Tuple, List[float]] = {}
        # Final per-cycle fit history for the cross-cycle consistency
        # check: name -> (map_value, posterior_std_fit_space, scale).
        # Mutually-incompatible confident fits across cycles are the
        # signature of an overconfident probe; flagged, and the verdict
        # downgraded, rather than silently trusted.
        self._sysid_fit_history: Dict[str, Tuple[float, float, str]] = {}
        # Agent-facing digest of the latest rollout fit (unexplainable
        # segments, unidentified/insensitive params, cross-cycle
        # conflicts); surfaced to the explorer as experiment objectives.
        self._last_sysid_diagnostics: str = ""

    @classmethod
    def get_name(cls) -> str:
        return "agent_sim_learning"

    # ── Predicate set ───────────────────────────────────────────

    def _get_all_predicates(self) -> Set[Predicate]:
        return self._kept_initial_predicates

    def _resolve_kept_names(self) -> Optional[FrozenSet[str]]:
        """Names of env predicates kept for the agent (None = keep all).

        The CFG flag overrides the class default.
        """
        cfg_override = getattr(CFG, "agent_sim_learn_kept_predicates_names",
                               None)
        if cfg_override:
            return frozenset(cfg_override)
        return self.KEPT_INITIAL_PREDICATE_NAMES

    def _compute_kept_initial_predicates(self) -> Set[Predicate]:
        """Apply the allowlist, then closure-strip derived predicates.

        A ``DerivedPredicate`` whose ``auxiliary_predicates`` reference
        any stripped predicate is itself stripped: keeping one with
        removed dependencies would expose a broken classifier to
        refinement.
        """
        kept_names = self._resolve_kept_names()
        if kept_names is None:
            return set(self._initial_predicates)
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

    # ── Agent session hooks ──────────────────────────────────────

    def _get_agent_system_prompt(self) -> str:
        if self._learning_mode:
            return self._build_synthesis_system_prompt()
        return super()._get_agent_system_prompt()

    def _get_synthesis_tool_names(self) -> Optional[List[str]]:
        """Complete tool surface for the synthesis agent.

        The names of the dynamic synthesis callables (just
        ``run_python``) attached to ``ctx.extra_mcp_tools`` inside
        :meth:`_synthesize_with_agent`. The mixin asserts the attached
        instances and this list agree. Fitting, residual reports, and
        plan validation are NOT tools: they live on the ``sim`` probe
        (``sim.fit`` / ``sim.residuals`` / ``sim.refine`` /
        ``sim.run``) inside ``run_python``.

        No inspect tools: the type/option digests are injected into the
        learn message (see :meth:`_build_synthesis_learn_message`) and
        trajectory access lives in ``run_python`` (``trajectories`` +
        ``describe_trajectory``). No ``explore_python`` either: in
        synthesis sessions the probe rides inside ``run_python``'s
        namespace as ``sim`` (one exec namespace per session - a helper
        defined next to the data is visible to probe sweeps). In the
        agent-synthesis session the probe runs against the CANDIDATE
        simulator.py via ctx.probe_option_model_provider (installed in
        _synthesize_with_agent); in the oracle-sim-program sampler
        session no provider is installed and the probe falls back to
        ctx.option_model, which there IS the deployed belief model.
        """
        names: List[str] = list(SYNTHESIS_TOOL_NAMES)
        # When the agent is learning samplers in this session (not using
        # ground-truth ones), expose the evaluate_sampler tool.
        if self._do_synthesize_samplers:
            names += list(SAMPLER_SYNTHESIS_TOOL_NAMES)
        return names

    # ── Subclass hooks ──────────────────────────────────────────
    # Default implementations are no-ops so subclasses can add
    # predicate-invention (or other) extensions without copying
    # _synthesize_with_agent.

    def _learning_cycle_index(self) -> int:
        """1-indexed cycle number used in versioned snapshot filenames.

        Offline learning is cycle 1; ``_online_learning_cycle`` is
        incremented before each online learn call, so adding 1 keeps the
        offline pass and the first online pass on different indices.
        """
        return self._online_learning_cycle + 1

    def _compute_extra_synthesis_paths(self, base: str) -> Dict[str, str]:
        """Return extra path bindings for the synthesis sandbox."""
        del base
        return {}

    def _extra_synthesis_tools(
        self,
        exec_ns: Dict[str, Any],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
        extra_paths: Dict[str, str],
    ) -> List[Any]:
        """Return additional MCP tools to append to the synthesis tool list."""
        del exec_ns, base_pred_triples, inferred_hint, extra_paths
        return []

    def _extra_synthesis_message(self, extra_paths: Dict[str, str]) -> str:
        """Return text to append to the agent's first synthesis message."""
        del extra_paths
        return ""

    def _extra_synthesis_system_prompt(self) -> str:
        """Return text to append to the synthesis system prompt."""
        return ""

    def _post_synthesis_loading(
        self,
        extra_paths: Dict[str, str],
        specs: List[ParamSpec],
    ) -> None:
        """Hook run after the simulator file is loaded post-session.

        ``specs`` are the just-loaded ``PARAM_SPECS``; subclasses may
        seed ``self._fitted_params`` from their ``init_value``s before
        the proper fit runs (useful when loading other artifacts that
        close over ``params``).
        """
        del extra_paths, specs

    def _build_write_snapshot_targets(
        self,
        simulator_file: str,
        versions_dir: str,
        extra_paths: Dict[str, str],
    ) -> List[_SnapshotTarget]:
        """Files the PostToolUse snapshot hook should watch.

        Defaults to just the simulator. Subclasses (e.g. predicate
        invention) may append their own artifacts. ``extra_paths`` is
        the same dict returned by ``_compute_extra_synthesis_paths``.
        """
        del extra_paths
        return [
            _SnapshotTarget(
                live_file=simulator_file,
                versions_dir=versions_dir,
                artifact_name="simulator",
                cycle_index_provider=self._learning_cycle_index,
            ),
        ]

    @staticmethod
    def _build_synthesis_session_hooks(
        targets: List[_SnapshotTarget],
        sandbox_dir: str,
    ) -> Dict[str, list]:
        """Wrap snapshot targets in a Claude Agent SDK ``HookMatcher``.

        Returns the dict suitable for assignment to
        ``ToolContext.extra_session_hooks``. Falls back to an empty dict
        if the SDK ``HookMatcher`` isn't importable (so the approach
        still works against older SDK versions).
        """
        if not targets:
            return {}
        try:
            from claude_agent_sdk import \
                HookMatcher  # pylint: disable=import-outside-toplevel
        except ImportError:
            logger.warning("claude_agent_sdk.HookMatcher unavailable; "
                           "write-time snapshots disabled.")
            return {}
        hook = make_write_snapshot_hook(targets, sandbox_dir=sandbox_dir)
        return {
            "PostToolUse": [
                HookMatcher(matcher="Write|Edit|MultiEdit", hooks=[hook]),
            ],
        }

    # ── Learning ────────────────────────────────────────────────

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        super().learn_from_offline_dataset(dataset)
        self._learn_simulator(self._get_all_trajectories())

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        super().learn_from_interaction_results(results)
        self._learn_simulator(self._get_all_trajectories())

    def _learn_simulator(self, trajectories: List[LowLevelTrajectory]) -> None:
        """Synthesize rules, fit parameters, and build the option model."""
        # Cache for recurrent fitting: lets _group_triples_by_trajectory
        # slice the flat base_pred_triples back into per-trajectory chunks
        # (latent threads within a trajectory, not across). Harmless for
        # fully-observable (legacy) simulators, which never regroup.
        self._fit_trajectories = list(trajectories)
        # New data invalidates the memoized explainability verdicts.
        self._explainability_cache.clear()
        # Decide how samplers are obtained this cycle: ground-truth (if
        # requested and available for the env) else agent synthesis. GT
        # samplers are static, so install them up front, independent of
        # whether simulator learning runs below (it is skipped when there
        # are no step transitions and no oracle sim program to fall
        # back on, e.g. when every demo failed).
        self._maybe_install_oracle_samplers()
        # Two parallel triple lists drive the rest of this method:
        # * obs_triples       - raw (s_t, a, s_{t+1}) from the data.
        # * base_pred_triples - same triples but s_t replaced by the
        #   base sim's one-step prediction. The rules run on top of that
        #   prediction; SSE compares against s_{t+1}.
        obs_triples = self._extract_obs_triples(trajectories)
        if not obs_triples and not CFG.agent_sim_learn_oracle_sim_program:
            logger.warning("No step transitions; skipping simulator learning.")
            return
        if obs_triples:
            # Headless env for the pre-compute: reusing the GUI base_env
            # corrupts its visual-shape state after a few hundred steps.
            fit_env = create_new_env(CFG.env,
                                     do_cache=False,
                                     use_gui=False,
                                     skip_process_dynamics=True)
            logger.info("Pre-computing base states for %d transitions.",
                        len(obs_triples))
            try:
                base_pred_triples = self._compute_base_pred_triples(
                    obs_triples, fit_env)
            finally:
                # This env is rebuilt every learning cycle; disconnect
                # its PyBullet client or each cycle leaks a full physics
                # world (~145MB for the domino env).
                pybullet.disconnect(fit_env._physics_client_id)  # type: ignore[attr-defined]  # pylint: disable=protected-access
            inferred_hint = self._infer_process_features_from_residuals(
                obs_triples, base_pred_triples)
            logger.info("Process features (data-driven hint): %s",
                        inferred_hint)
        else:
            # The oracle sim program is data-free (rules and parameter
            # inits come from get_gt_simulator), so a run whose every
            # demo failed still gets a working option model; the fit
            # below degrades to the declared inits.
            logger.warning("No step transitions; loading oracle sim "
                           "program without data.")
            base_pred_triples = []
            inferred_hint = {}

        self._synthesize_with_agent(trajectories, obs_triples,
                                    base_pred_triples, inferred_hint)

        if self._process_rules is not None and self._fitted_params:
            rules, params = self._process_rules, self._fitted_params
            self._learned_simulator = LearnedSimulator(
                step_fn=lambda s, _r=rules, _p=params:  # type: ignore[misc]
                apply_rules(s, _r, _p),
                name="agent_synthesized")
        elif self._learned_simulator is None:
            logger.warning("Synthesis produced no simulator, skipping.")
            return

        combined_sim = self._build_combined_simulator(self._learned_simulator)
        self._option_model = self._build_option_model(combined_sim)
        logger.info("Built learned option model (SSE: %.6f).", self._fit_sse)

        # When the simulator came from the oracle short-circuit no agent
        # session ran above, so per-skill samplers (if enabled) get their
        # own session here, after the option model is built so the
        # session's probe (sim.refine) has a working simulator. When
        # the agent *did* synthesize the simulator, samplers already rode
        # along in that session and this is skipped.
        if self._do_synthesize_samplers and \
                CFG.agent_sim_learn_oracle_sim_program:
            if base_pred_triples:
                self._synthesize_samplers_standalone(trajectories,
                                                     base_pred_triples,
                                                     inferred_hint)
            else:
                logger.warning("No step transitions; skipping standalone "
                               "sampler synthesis.")

    def _build_option_model(
        self,
        simulator_fn: Callable[[State, Action], State],
    ) -> _OracleOptionModel:
        """Wrap a simulator function in an OracleOptionModel.

        Uses ``self._get_all_options()`` rather than
        ``get_gt_options(CFG.env)`` to avoid spawning a second cached
        PyBullet env via ``get_or_create_env``.
        """
        model = _OracleOptionModel(self._get_all_options(), simulator_fn)
        # The learned simulator_fn rides on top of _base_env's physics,
        # so that env is the one physics-needing task-evaluator
        # certificates (the domino counterfactual push probe) must run
        # against. Without this the probe is silently unavailable in the
        # sandbox and captures are accepted on the pure rules only.
        model.sim_env = self._base_env
        if CFG.wait_option_terminate_on_atom_change:
            model._abstract_function = (  # pylint: disable=protected-access
                lambda s: utils.abstract(s, self._get_all_predicates()))
        return model

    def _make_candidate_probe_model_provider(
        self,
        simulator_file: str,
        trajectories: List[LowLevelTrajectory],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
    ) -> Callable[[], _OracleOptionModel]:
        """Lazy option-model builder behind the synthesis explore_python.

        The returned callable is installed as
        ``ctx.probe_option_model_provider`` for the synthesis session:
        on first use, and again after every content change of
        ``simulator_file``, it loads the candidate simulator, MCMC-fits
        its params, and builds the combined option model through the
        :func:`build_candidate_option_model` path, publishing the fit
        exactly as ``sim.fit`` reports it. ``sim.run`` / ``sim.refine``
        therefore always exercise the candidate at deployed (fitted)
        params. Content-hash caching keeps
        sweep loops cheap: an unchanged file never refits.

        Raises ``RuntimeError`` (surfaced in the tool output) when no
        loadable candidate exists yet - the probe must never fall back
        to the pre-synthesis option model, which on cycle 1 wraps the
        real env (a live-physics leak into learning).
        """
        cache: Dict[str, Any] = {}

        def _provider() -> _OracleOptionModel:
            if not os.path.isfile(simulator_file):
                raise RuntimeError(
                    "explore_python probe: no candidate simulator yet - "
                    "write ./simulator.py (PROCESS_RULES / PARAM_SPECS / "
                    "PROCESS_FEATURES) first; the probe runs against it.")
            with open(simulator_file, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if cache.get("digest") == digest:
                return cache["model"]
            rules, specs, features, ns = \
                self._load_simulator_from_module_file(
                    simulator_file, trajectories)
            if rules is None or specs is None:
                raise RuntimeError(
                    "explore_python probe: ./simulator.py failed to load "
                    "(exec error, or PROCESS_RULES / PARAM_SPECS missing) - "
                    "fix the file and probe again.")
            process_features = (features
                                if features is not None else inferred_hint)
            latent_init = read_latent_init(ns) if isinstance(ns,
                                                             dict) else None
            model, _, fit_sse = build_candidate_option_model(
                self,
                rules,
                specs,
                process_features,
                base_pred_triples,
                latent_init=latent_init)
            logger.info(
                "Synthesis probe: candidate model rebuilt from %s "
                "(post-fit SSE %.6f).", simulator_file, fit_sse)
            cache["digest"] = digest
            cache["model"] = model
            return model

        return _provider

    # ── Active-experiment ensemble (info-seeking exploration) ────

    @staticmethod
    def _exploration_fit_num_steps() -> Optional[int]:
        """MCMC budget for the active-experiment posterior fit.

        The synthesis fit surfaces (``sim.fit``, ``sim.residuals``)
        share the fit statics and run repeatedly inside the agent loop,
        so they always use the global
        ``CFG.code_sim_learning_num_mcmc_steps`` (typically 0 - LM +
        Laplace only). The solver/test-time fit also uses that global
        setting. The exploration posterior fit is different: it runs
        once per learning cycle, only when it needs more MCMC than the
        solver fit already ran, and its posterior feeds only the
        info-seeking ensemble. With real posterior samples,
        ``_select_param_ensemble`` upgrades from the Laplace draw to a
        posterior subsample, calibrating ensemble spread for
        gate/threshold params whose flat likelihood has a near-zero
        Jacobian column at the MAP (invisible to Laplace).

        Returns ``None`` (no override; ``fit_params`` falls back to the
        global setting) when info-seeking is off, else the max of the
        global and exploration budgets so the override never *reduces*
        an explicitly configured global MCMC run.
        """
        if not CFG.agent_explorer_info_seeking:
            return None
        return max(CFG.code_sim_learning_num_mcmc_steps,
                   CFG.agent_explorer_info_mcmc_steps)

    @staticmethod
    def _separate_exploration_fit_num_steps() -> Optional[int]:
        """Return an exploration-only MCMC budget, if one is needed."""
        fit_num_steps = AgentSimLearningApproach._exploration_fit_num_steps()
        if fit_num_steps is None:
            return None
        if fit_num_steps <= CFG.code_sim_learning_num_mcmc_steps:
            return None
        return fit_num_steps

    def _rebuild_param_ensemble(self) -> None:
        """Rebuild the active-experiment parameter ensemble.

        No-op (clears the ensemble) unless info-seeking exploration is
        enabled and a fit has populated ``_fitted_params``. The ensemble
        can use an exploration-only posterior even when solver params
        remain at the global-budget point estimate.

        Picks the most *calibrated* ensemble the fit affords, preferring
        spreads that reflect real posterior uncertainty over uniform
        jitter (see :meth:`_select_param_ensemble`).
        """
        if (not CFG.agent_explorer_info_seeking or not self._fitted_params):
            self._param_ensemble = []
            return
        num_members = CFG.agent_explorer_info_ensemble_size
        self._param_ensemble, method = self._select_param_ensemble(num_members)
        logger.info(
            "Built active-experiment ensemble: %d members via %s over "
            "%d params.", len(self._param_ensemble), method,
            len(self._param_specs))

    def _select_param_ensemble(
            self, num_members: int) -> Tuple[List[Dict[str, float]], str]:
        """Choose and build the ensemble, returning (members, method-label).

        Dispatch, most- to least-calibrated:

        * ``posterior`` - when MCMC ran (``num_mcmc_steps > 0``), subsample
          the real posterior ``samples`` (works for both per-transition and
          recurrent fits).
        * ``laplace`` - else, when the fit attached an LM Jacobian
          (``num_mcmc_steps == 0``, per-transition or recurrent), draw
          from the Laplace covariance at the MAP.
        * ``uniform`` - otherwise (oracle params, LM skipped/failed, or
          calibration disabled), fall back to box-relative jitter.
        """
        fit = self._last_fit_result
        calibrated = CFG.agent_explorer_info_calibrated_ensemble
        if calibrated and fit is not None:
            samples = np.asarray(fit.samples, dtype=float)
            if samples.ndim == 2 and samples.shape[0] > 1:
                return posterior_subsample_ensemble(
                    fit.point_estimate,
                    fit.names,
                    samples,
                    num_members=num_members,
                    rng=self._rng,
                ), "posterior-subsample"
            if (fit.jacobian is not None and fit.noise_sigma is not None
                    and fit.prior_sigma is not None):
                return laplace_ensemble(
                    self._fitted_params,
                    fit.names,
                    self._param_specs,
                    fit.jacobian,
                    fit.noise_sigma,
                    fit.prior_sigma,
                    num_members=num_members,
                    rng=self._rng,
                ), "laplace"
        return perturbation_ensemble(
            self._fitted_params,
            self._param_specs,
            num_members=num_members,
            perturb_frac=CFG.agent_explorer_info_perturb_frac,
            rng=self._rng,
        ), "uniform-perturb"

    def score_atom_disagreement(self, state: State,
                                atoms: Collection[GroundAtom]) -> float:
        """Ensemble disagreement (mean Bernoulli entropy) over ``atoms``.

        Evaluates each atom's truth in ``state`` under every ensemble
        member by swapping ``_fitted_params`` (which the learned
        predicate classifiers read through ``_ParamsView``) to each
        member in turn, then restoring it. High disagreement marks a
        state that straddles a learned predicate's decision boundary,
        i.e. an informative experiment. Returns 0.0 when the ensemble is
        trivial (<=1 member) or no atoms are given.

        Wired into refinement as the info-scorer for the agent_bilevel
        explorer; a read-only query that leaves ``_fitted_params``
        unchanged on return.
        """
        atom_list = list(atoms)
        if len(self._param_ensemble) <= 1 or not atom_list:
            return 0.0
        saved = dict(self._fitted_params)
        try:
            rows: List[List[bool]] = []
            for member in self._param_ensemble:
                self._fitted_params.clear()
                self._fitted_params.update(member)
                rows.append([bool(a.holds(state)) for a in atom_list])
        finally:
            self._fitted_params.clear()
            self._fitted_params.update(saved)
        return mean_bernoulli_entropy(np.asarray(rows, dtype=bool))

    # ── Agent-based synthesis ────────────────────────────────────

    def _synthesize_with_agent(
        self,
        trajectories: List[LowLevelTrajectory],
        obs_triples: List[Tuple[State, Action, State]],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
    ) -> None:
        """Obtain PROCESS_RULES / PARAM_SPECS / PROCESS_FEATURES, then fit.

        ``inferred_hint`` is passed to the agent as a starting point and
        used as the eval/test scope until it declares its own
        ``PROCESS_FEATURES``. CFG flag
        ``agent_sim_learn_oracle_sim_program`` short-circuits the agent
        session by loading the GT simulator instead (and
        ``agent_sim_learn_oracle_sim_params`` additionally skips the
        MCMC fit; see :meth:`_fit_params_after_synthesis`).
        """
        if CFG.agent_sim_learn_oracle_sim_program:
            rules, specs, process_features = \
                self._load_oracle_sim_program(inferred_hint)
        else:
            loaded = self._run_agent_synthesis_session(trajectories,
                                                       obs_triples,
                                                       base_pred_triples,
                                                       inferred_hint)
            if loaded is None:
                return
            rules, specs, process_features = loaded
        self._process_rules = rules
        self._process_features = process_features
        self._fit_params_after_synthesis(rules, specs, base_pred_triples,
                                         process_features)

    def _load_oracle_sim_program(
        self, inferred_hint: Dict[str, List[str]]
    ) -> Tuple[List, List[ParamSpec], Dict[str, List[str]]]:
        """Load the ground-truth simulator instead of running an agent.

        ``get_gt_simulator`` dispatches by observability: in
        partially-observable mode it returns the PO GT simulator
        (gt_simulator_po.py - latent heat threaded across steps,
        surfaced as the observable bubbling_level), which predicts only
        observable features; otherwise it returns the fully-observable
        gt_simulator.py (which reads/writes heat_level as a State
        feature). The two factories gate on CFG.partially_observable so
        the env-name dispatch resolves to exactly one module per run.

        Unless ``agent_sim_learn_oracle_sim_params`` also holds, the
        declared parameter inits are perturbed so the subsequent fit
        starts from a miscalibrated - not oracle - belief.
        """
        rules, specs, process_features = get_gt_simulator(CFG.env)
        self._log_feature_set_diff(inferred_hint, process_features, "inferred",
                                   "oracle")
        if not CFG.agent_sim_learn_oracle_sim_params:
            specs = self._perturb_spec_inits(specs)
        logger.info("Loaded oracle sim program (%d rules, %d params).",
                    len(rules), len(specs))
        return rules, specs, process_features

    @staticmethod
    def _perturb_spec_inits(specs: List[ParamSpec]) -> List[ParamSpec]:
        """Perturb each spec's init with multiplicative Gaussian noise.

        Used when the oracle sim PROGRAM is loaded but its param VALUES
        must still be learned: the fit then starts from a plausible but
        wrong belief instead of the answer. Each perturbed init is
        clipped to its spec's box.
        """
        rng = np.random.default_rng(CFG.seed)
        noise_scale = CFG.agent_sim_learn_oracle_sim_param_noise_scale
        if noise_scale < 0.0:
            raise ValueError("agent_sim_learn_oracle_sim_param_noise_scale "
                             "must be non-negative.")
        perturbed = []
        for s in specs:
            val = float(
                np.clip(s.init_value * (1.0 + rng.normal(0, noise_scale)),
                        s.lo, s.hi))
            perturbed.append(
                ParamSpec(s.name,
                          val,
                          lo=s.lo,
                          hi=s.hi,
                          scale=getattr(s, "scale", "linear")))
        return perturbed

    def _run_agent_synthesis_session(
        self,
        trajectories: List[LowLevelTrajectory],
        obs_triples: List[Tuple[State, Action, State]],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
    ) -> Optional[Tuple[List, List[ParamSpec], Dict[str, List[str]]]]:
        """Run one agent synthesis session and load what it committed.

        Returns ``(rules, specs, process_features)``, or None when the
        session left no loadable simulator artifact. Per-skill samplers
        (when enabled) ride along in the same session.
        """
        paths = self._resolve_synthesis_paths()
        extra_paths = self._compute_extra_synthesis_paths(paths.base)
        sampler_paths = (self._sampler_paths(paths.base)
                         if self._do_synthesize_samplers else {})
        exec_ns = self._build_synthesis_exec_ns(trajectories)
        self._attach_synthesis_session_state(exec_ns, trajectories,
                                             base_pred_triples, inferred_hint,
                                             paths, extra_paths, sampler_paths)
        # Fresh session so the synthesis prompt + tools take effect.
        self._close_agent_session()
        self._ensure_agent_session()
        structs_ref = self._write_structs_reference()
        message = self._build_synthesis_learn_message(trajectories,
                                                      obs_triples,
                                                      inferred_hint, paths,
                                                      structs_ref, extra_paths,
                                                      sampler_paths)
        try:
            self._query_agent_sync(message, kind="learn")
        finally:
            self._tool_context.extra_session_hooks = {}
            self._tool_context.extra_mcp_tools = []
            self._tool_context.probe_option_model_provider = None
            self._tool_context.probe_fit_provider = None
            self._tool_context.probe_residuals_provider = None
            self._learning_mode = False
            self._close_agent_session()
        return self._load_synthesis_artifacts(trajectories, inferred_hint,
                                              paths, extra_paths,
                                              sampler_paths)

    def _resolve_synthesis_paths(self) -> _SynthesisPaths:
        """Host- and agent-visible paths for one synthesis session.

        The sandbox dir is resolved without depending on a live session
        manager: LocalSandboxSessionManager does set it on tool_context
        in __init__, but it isn't constructed until
        ``_ensure_agent_session()`` runs later in the session setup.

        The agent-visible paths differ by sandbox backend: cwd-relative
        for local-sandbox (the validation hook resolves against cwd and
        rejects literal ``/sandbox/...`` paths), the docker mount point
        for docker, the absolute host path otherwise.
        """
        if CFG.agent_sdk_use_local_sandbox:
            sandbox_dir: Optional[str] = os.path.abspath(
                os.path.join(self._get_log_dir(), "sandbox"))
        else:
            sandbox_dir = self._tool_context.sandbox_dir
        base = sandbox_dir or self._get_log_dir()
        simulator_file = os.path.join(base, "simulator.py")
        if CFG.agent_sdk_use_local_sandbox:
            simulator_file_for_agent = "./simulator.py"
            sandbox_dir_for_agent: Optional[str] = "."
        elif sandbox_dir:
            simulator_file_for_agent = "/sandbox/simulator.py"
            sandbox_dir_for_agent = "/sandbox"
        else:
            simulator_file_for_agent = simulator_file
            sandbox_dir_for_agent = None
        return _SynthesisPaths(
            base=base,
            simulator_file=simulator_file,
            versions_dir=os.path.join(base, "simulator_versions"),
            simulator_file_for_agent=simulator_file_for_agent,
            sandbox_dir_for_agent=sandbox_dir_for_agent)

    def _build_synthesis_exec_ns(
            self, trajectories: List[LowLevelTrajectory]) -> Dict[str, Any]:
        """Variables exposed to the synthesis agent's ``run_python``."""
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
        # Curated per-trajectory digest (same renderer the old
        # inspect_trajectories tool used), for a first look before
        # ad-hoc exploration over the raw ``trajectories`` objects.
        all_predicates = self._get_all_predicates()

        def describe_trajectory(traj_idx: int,
                                include_states: bool = True,
                                include_atoms: bool = False,
                                max_timesteps: int = 10) -> str:
            return render_trajectory_digest(trajectories,
                                            self._train_tasks,
                                            all_predicates,
                                            traj_idx,
                                            include_states=include_states,
                                            include_atoms=include_atoms,
                                            max_timesteps=max_timesteps)

        exec_ns["describe_trajectory"] = describe_trajectory
        # Env ground-truth scoring, next to is_goal_state (see
        # Task.evaluator). Verdict-only surface: dict of reward/solved
        # on a concrete state sequence - real trajectories or the
        # agent's own simulator rollouts (there the verdict is only as
        # good as the sim).
        if any(t.evaluator is not None for t in self._train_tasks):
            exec_ns["evaluate_trajectory"] = \
                self._make_evaluate_trajectory_fn()
        return exec_ns

    def _attach_synthesis_session_state(
        self,
        exec_ns: Dict[str, Any],
        trajectories: List[LowLevelTrajectory],
        base_pred_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
        paths: _SynthesisPaths,
        extra_paths: Dict[str, str],
        sampler_paths: Dict[str, str],
    ) -> None:
        """Install this synthesis session's state on the tool context.

        Everything installed here is cleared by the caller's ``finally``
        once the session query returns.
        """
        # Build dynamic synthesis tools and attach them to the tool
        # context *before* opening the session. The attached set is
        # filtered against ``_get_synthesis_tool_names`` so that method
        # is the single source of truth for what the agent sees:
        # anything a builder constructs but the names list omits is
        # dropped here.
        toolkit = create_synthesis_tools(
            exec_ns,
            base_pred_triples,
            inferred_hint,
            simulator_file=paths.simulator_file,
            versions_dir=paths.versions_dir,
            approach=self,
            sandbox_dir=paths.base,
            sandbox_dir_for_agent=paths.sandbox_dir_for_agent,
            cycle_index_provider=self._learning_cycle_index,
        )
        tools = list(toolkit.tools)
        tools.extend(
            self._extra_synthesis_tools(exec_ns, base_pred_triples,
                                        inferred_hint, extra_paths))
        if self._do_synthesize_samplers:
            tools.extend(self._make_sampler_tools(sampler_paths))
        declared = set(self._get_synthesis_tool_names() or ())
        self._tool_context.extra_mcp_tools = [
            t for t in tools if getattr(t, "name", "") in declared
        ]
        # Point the probe at the CANDIDATE simulator for this session
        # (never the stale pre-synthesis option model; on cycle 1 that
        # wraps the real env), then merge the probe facade into
        # run_python's namespace: synthesis sessions offer ONE exec
        # namespace, so helpers defined next to the data are visible to
        # probe sweeps (no explore_python tool here - the roster method
        # documents the policy). Unconditional: with fit / refine /
        # forward-validation all living on ``sim``, the probe IS the
        # validation surface, so a synthesis session without it would
        # have no way to test what it writes. Only ``sim``/``ProbeSim``
        # are taken from the probe namespace: ``trajectories`` already
        # binds the fit list and solve-only extras do not apply.
        self._tool_context.probe_option_model_provider = \
            self._make_candidate_probe_model_provider(
                paths.simulator_file, trajectories, base_pred_triples,
                inferred_hint)
        self._tool_context.probe_fit_provider = toolkit.fit_runner
        self._tool_context.probe_residuals_provider = \
            toolkit.residuals_runner
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.probe_api import build_probe_namespace
        probe_ns = build_probe_namespace(self._tool_context)
        exec_ns["sim"] = probe_ns["sim"]
        exec_ns["ProbeSim"] = probe_ns["ProbeSim"]
        self._learning_mode = True
        # PostToolUse hook: snapshot simulator.py / predicates.py on
        # every successful Write/Edit/MultiEdit, so the version history
        # covers everything the agent committed to file (not just
        # states that happened to coincide with an eval call). Only
        # active for this synthesis session.
        snapshot_targets = self._build_write_snapshot_targets(
            paths.simulator_file, paths.versions_dir, extra_paths)
        if self._do_synthesize_samplers:
            snapshot_targets.append(
                self._sampler_snapshot_target(sampler_paths))
        self._tool_context.extra_session_hooks = (
            self._build_synthesis_session_hooks(snapshot_targets, paths.base))

    def _build_synthesis_learn_message(
        self,
        trajectories: List[LowLevelTrajectory],
        obs_triples: List[Tuple[State, Action, State]],
        inferred_hint: Dict[str, List[str]],
        paths: _SynthesisPaths,
        structs_ref: str,
        extra_paths: Dict[str, str],
        sampler_paths: Dict[str, str],
    ) -> str:
        """Compose the synthesis session's first user message.

        Reads the just-opened session's tool names, so the session must
        be open before this is called.
        """
        n_trajs = len(trajectories)
        n_demos = sum(1 for t in trajectories if t.is_demo)
        n_interaction = n_trajs - n_demos
        predicate_listing = self._format_predicate_signatures(
            self._get_all_predicates())
        # Static per-session digests, injected instead of offering the
        # inspect_types / inspect_options tools (same renderers, zero
        # turns; see the roster note in _get_synthesis_tool_names).
        types_digest = render_types_digest(self._tool_context.types)
        options_digest = render_options_digest(
            self._tool_context.options,
            gt_options_ref_path=self._tool_context.gt_options_ref_path)
        trajectory_listing = self._format_trajectory_listing(trajectories)
        prior_state_block = self._format_prior_state_block(paths.base)
        objective_block = self._format_objective_block()
        simulator_file_for_agent = paths.simulator_file_for_agent
        # The probe rides inside run_python's namespace (one exec
        # namespace per session): `trajectories` is the recorded DATA,
        # `sim` forward-rolls the CANDIDATE simulator. One sentence so
        # the two are not conflated; details live in the tool
        # description.
        probe_note = ""
        if CFG.agent_planner_use_explore_python:
            probe_note = (
                "\n\nTo forward-roll your CURRENT candidate simulator "
                "(does the process fire when and where you intended?), "
                "use the `sim` probe available inside `run_python`: it "
                "probes the `simulator.py` you are editing, re-fit "
                "automatically whenever the file changes; pass task_idx "
                "explicitly to `sim.reset` (and `sim.task(task_idx)` for "
                "a task digest). Its rollouts are CANDIDATE-simulator "
                "predictions - do not mix them up with the recorded real "
                "`trajectories`. Exploratory only - "
                "validate via `sim.fit()` + `sim.refine` + a "
                "continuous `sim.run` forward pass before declaring "
                "the simulator done.")
        # Tool surface of the (just-opened) synthesis session, rendered
        # the same way the solve/explore prompts list theirs.
        # ``tool_names`` already merges the sandbox built-ins with the
        # declared MCP subset, prefix-stripped.
        tools_block = ""
        session_tool_names = (self._agent_session.tool_names
                              if self._agent_session is not None else [])
        if session_tool_names:
            tool_listing = "\n".join(f"  - {t}" for t in session_tool_names)
            tools_block = f"## Available Tools\n{tool_listing}\n\n"
        message = f"""\
Synthesize a process dynamics simulator for this environment. \
There are {n_trajs} trajectories ({len(obs_triples)} step \
transitions) available: {n_demos} oracle demonstration(s) (goal \
reached by construction) and {n_interaction} interaction \
trajectory/ies (collected during online learning; some may have \
failed to reach the goal).

{trajectory_listing}
Each trajectory carries a `train_task_idx`. You can query the \
ground-truth goal-atom check by calling \
`is_goal_state(state, task_idx)`. Equivalently \
`train_tasks[task_idx].goal_holds(state)`. This checks a single \
STATE for the goal atoms only - reaching the goal atoms does not by \
itself mean an episode is solved; when a task objective is stated \
below, score full trajectories with `evaluate_trajectory`. Use \
`is_goal_state` to (1) confirm which trajectories reached the goal \
atoms and (2) treat failed interaction trajectories as \
counterexamples - places where your predicate or rule said "this \
should work" but the env disagreed.

{objective_block}{prior_state_block}Data-structure source code is at: \
{structs_ref}

A residual scan between the base simulator's prediction and the \
observed next state suggests these features carry process dynamics \
(starting hint, may include base-sim jitter - refine as you go):
{inferred_hint}

## Available Predicates (for subgoal annotations)
{predicate_listing}

Subgoal annotations in your plans for `sim.refine` / `sim.run` \
must reference these predicate names with matching arity and types. \
Any threshold or condition you bake into a rule must be consistent \
with what the predicate's classifier actually checks, or refinement \
will reject parameter samples that look correct on paper.

## Object Types
{types_digest}

## Options
Plans (for `sim.refine` / `sim.run`) and rules must match these \
typed signatures and parameter boxes exactly:
{options_digest}

{tools_block}Read the data-structures file first, then explore the trajectory \
data with `run_python` (variables: `trajectories`, `train_tasks`, \
`is_goal_state`, `describe_trajectory(traj_idx)` for a per-timestep \
digest, `np`, `ParamSpec`, plus `evaluate_trajectory` when a \
task objective is stated above). Write your simulator to \
`{simulator_file_for_agent}` - define PROCESS_RULES, PARAM_SPECS, \
and PROCESS_FEATURES there. Every successful Write/Edit of \
`{simulator_file_for_agent}` is snapshotted to `simulator_versions/` as \
`cycle_XXX_vers_YYY_simulator.py` (deduped by content); `sim.fit` and \
`sim.residuals` \
load that file fresh on every call and report the version tag \
[cycle_XXX_vers_YYY] in their output. Iterate with `Edit` and \
re-score.{probe_note}"""

        extra_message = self._extra_synthesis_message(extra_paths)
        if extra_message:
            message = message + "\n\n" + extra_message
        if self._do_synthesize_samplers:
            message = message + "\n\n" + \
                self._sampler_synthesis_message(sampler_paths)
        return message

    def _load_synthesis_artifacts(
        self,
        trajectories: List[LowLevelTrajectory],
        inferred_hint: Dict[str, List[str]],
        paths: _SynthesisPaths,
        extra_paths: Dict[str, str],
        sampler_paths: Dict[str, str],
    ) -> Optional[Tuple[List, List[ParamSpec], Dict[str, List[str]]]]:
        """Load the artifacts the finished session committed to disk.

        Returns ``(rules, specs, process_features)`` or None when no
        loadable simulator exists. The optional LATENT_INIT /
        PHYSICAL_PARAMS side exports are recorded on ``self`` before the
        loadability check, so they are picked up even from an artifact
        whose rules fail to load.
        """
        final_sim_tag = finalize_versioned_snapshot(
            paths.simulator_file,
            paths.versions_dir,
            cycle_idx=self._learning_cycle_index(),
            artifact_name="simulator",
        )
        if final_sim_tag is not None:
            self._current_simulator_version = final_sim_tag
            logger.info("Final simulator snapshot: %s", final_sim_tag)

        rules, specs, declared_features, sim_ns = (
            self._load_simulator_from_module_file(paths.simulator_file,
                                                  trajectories))
        # Pick up the optional LATENT_INIT export (partial
        # observability). None for fully-observable simulators, which
        # leaves every latent path dormant.
        self._latent_init = (read_latent_init(sim_ns) if isinstance(
            sim_ns, dict) else None)
        # Optional PHYSICAL_PARAMS export: base-sim parameters to
        # identify jointly with the rule params (system ID). The fit
        # scale (log vs linear) is stamped from the env registry; agents
        # copy name/init/bounds but need not know about it.
        self._physical_param_specs = stamp_physical_spec_scales(
            list((read_physical_param_specs(sim_ns) if isinstance(
                sim_ns, dict) else None) or []), self._base_env)
        if self._physical_param_specs:
            logger.info("Agent declared %d physical params for system ID: %s",
                        len(self._physical_param_specs),
                        [s.name for s in self._physical_param_specs])
        if rules is None or specs is None:
            return None
        assert declared_features is not None, (
            "Agent did not declare PROCESS_FEATURES; "
            "synthesis output is incomplete.")
        process_features = declared_features
        self._log_feature_set_diff(inferred_hint, process_features, "inferred",
                                   "declared")
        logger.info("Agent synthesized %d rules, %d params.", len(rules),
                    len(specs))
        self._post_synthesis_loading(extra_paths, specs)
        if self._do_synthesize_samplers:
            self._finalize_and_load_samplers(sampler_paths)
        return rules, specs, process_features

    def _fit_params_after_synthesis(
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        process_features: Dict[str, List[str]],
    ) -> None:
        """Fit/store solver params and, separately, explorer posterior."""
        if CFG.agent_sim_learn_oracle_sim_params:
            self._fitted_params.clear()
            self._fitted_params.update({s.name: s.init_value for s in specs})
            if self._physical_param_specs:
                # Oracle mode: trust the agent-declared physical inits.
                self._apply_identified_physical_params(
                    {s.name: s.init_value
                     for s in self._physical_param_specs})
            # No fit ran; the ensemble falls back to uniform perturbation.
            self._last_fit_result = None
            if base_pred_triples:
                self._fit_sse = self._oracle_param_sse(rules,
                                                       base_pred_triples,
                                                       process_features,
                                                       FIT_NOISE_SIGMA)
            else:
                logger.info("No transitions; skipping oracle-param SSE.")
                self._fit_sse = float("inf")
        elif not base_pred_triples:
            # No data to fit against (e.g. every demo failed): seed from
            # the declared inits so the simulator still builds; later
            # cycles refit once transitions arrive.
            logger.warning("No transitions to fit; seeding params from "
                           "declared inits.")
            self._fitted_params.clear()
            self._fitted_params.update({s.name: s.init_value for s in specs})
            self._last_fit_result = None
            self._fit_sse = float("inf")
        else:
            # This is the solver/test-time fit. It deliberately follows
            # CFG.code_sim_learning_num_mcmc_steps; any extra
            # info-seeking MCMC is run below and is not published into
            # _fitted_params.
            if self._physical_param_specs:
                # System ID: physical + rule params fit jointly against
                # free-running rollouts (teacher-forced triples cannot
                # see physical params - no velocities in State).
                fit_result, self._fit_sse = (
                    self._fit_parameters_joint_rollout(rules, specs,
                                                       process_features))
            elif has_latent_rules(rules):
                fit_result, self._fit_sse = self._fit_parameters_recurrent(
                    rules, specs, base_pred_triples, process_features)
            else:
                fit_result, self._fit_sse = fit_rule_parameters(
                    rules, specs, base_pred_triples, process_features)
            self._last_fit_result = fit_result
            self._fitted_params.clear()
            self._fitted_params.update(fit_result.point_estimate)
            if CFG.code_sim_learning_num_mcmc_steps == 0:
                logger.info("Skipped solver MCMC; using %d fitted params.",
                            len(specs))
            else:
                logger.info("Fitted %d solver params.", len(specs))

            self._maybe_refit_exploration_posterior(rules, specs,
                                                    base_pred_triples,
                                                    process_features)

        # Remember the specs (names + bounds) and rebuild the active-
        # experiment ensemble. Cheap and only consumed when info-seeking
        # exploration is enabled. Physical specs lead so the ordering
        # matches the joint rollout fit's theta layout.
        self._param_specs = list(self._physical_param_specs) + list(specs)
        self._rebuild_param_ensemble()

    def _maybe_refit_exploration_posterior(
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        process_features: Dict[str, List[str]],
    ) -> None:
        """Run the exploration-only posterior fit, when one is needed.

        A no-op unless info-seeking exploration asks for more MCMC than
        the solver fit already ran (see
        :meth:`_separate_exploration_fit_num_steps`). The resulting
        posterior replaces ``_last_fit_result`` for ensemble calibration
        only; ``_fitted_params`` (the solver's point estimate) is left
        untouched.
        """
        num_steps = self._separate_exploration_fit_num_steps()
        if num_steps is None:
            return
        if self._physical_param_specs:
            logger.info("Skipping separate active-experiment fit: the joint "
                        "rollout sysID posterior is reused for exploration.")
            return
        if has_latent_rules(rules):
            fit_result, sse = self._fit_parameters_recurrent(
                rules,
                specs,
                base_pred_triples,
                process_features,
                num_steps=num_steps)
        else:
            fit_result, sse = fit_rule_parameters(rules,
                                                  specs,
                                                  base_pred_triples,
                                                  process_features,
                                                  num_steps=num_steps)
        self._last_fit_result = fit_result
        logger.info(
            "Fitted active-experiment posterior with %d MCMC steps "
            "for exploration planning only (SSE: %.6f).", num_steps, sse)

    # ── Parameter fitting ────────────────────────────────────────

    def _oracle_param_sse(
        self,
        rules: List,
        base_pred_triples: List[Tuple[State, Action, State]],
        process_features: Dict[str, List[str]],
        noise_sigma: float,
    ) -> float:
        """Compute and log the SSE for oracle params (no fitting).

        ``self._fitted_params`` is assumed already populated with the
        oracle values. Returns the SSE. Recurrent (5-arg) rules cannot
        run per-transition, so when the loaded rules carry a latent
        block this dispatches to :meth:`_oracle_param_sse_recurrent`;
        otherwise it rolls each transition independently through the
        legacy 3-arg ``apply_rules``.
        """
        if has_latent_rules(rules):
            return self._oracle_param_sse_recurrent(rules, base_pred_triples,
                                                    process_features,
                                                    noise_sigma)
        oracle_sim_fn = lambda s, a, p: apply_rules(  # noqa: E731
            s, rules, p)
        sse = compute_sse(oracle_sim_fn, base_pred_triples,
                          self._fitted_params, process_features)
        fit_ll = -0.5 * sse / (noise_sigma**2)
        logger.info("Oracle params - SSE: %.6f  log-likelihood: %.2f", sse,
                    fit_ll)
        for name, val in sorted(self._fitted_params.items()):
            logger.info("  %-30s  %.4f", name, val)
        log_sse_breakdown(oracle_sim_fn,
                          base_pred_triples,
                          self._fitted_params,
                          process_features,
                          label="oracle")
        return sse

    # ── System identification (PHYSICAL_PARAMS) support ──────────

    def _get_rollout_fit_env(self) -> Any:
        """Factory for the headless envs the rollout fit rolls out in.

        Returns a zero-arg callable; ``rollout_states`` invokes it once
        per rollout and disconnects the fresh env's PyBullet client
        afterwards. A fresh DIRECT-mode world per rollout is required
        for the fit to be deterministic at all: on a reused env the same
        theta produced SSE alternating 0.15/78 (run_20260708_213258),
        corrupting the grid seed and flooring the identifiability
        probe's same-theta noise floor - state-level resets cannot flush
        PyBullet solver internals (see ``rollout_states``). Measured
        overhead ~0.15 s per rollout on the domino env. It also keeps
        the fit's dynamics mutations away from the planning
        ``self._base_env`` (whose GUI variant additionally corrupts
        visual-shape state after a few hundred steps).
        """

        def _make() -> Any:
            return create_new_env(CFG.env,
                                  do_cache=False,
                                  use_gui=False,
                                  skip_process_dynamics=True)

        return _make

    def _rollout_fit_trajectories(
        self,
        process_features: Optional[Dict[str, List[str]]] = None,
    ) -> List[RolloutTrajectory]:
        """Raw observed (states, actions) sequences for rollout matching.

        Unlike ``base_pred_triples`` these keep each trajectory whole, so
        momentum can accrue across steps in the free-running rollout.
        When ``process_features`` is given (the fit's scored features)
        and ``CFG.code_sim_learning_rollout_truncate_settled`` is on,
        each trajectory's static tail is cut (see
        :func:`trajectory_prep.truncate_settled_tail`) so the fit scores
        the active cascade, not hundreds of settled steps of accumulated
        rollout divergence.
        """
        rollouts: List[RolloutTrajectory] = []
        for traj in self._fit_trajectories:
            if traj.actions and len(traj.states) == len(traj.actions) + 1:
                rollouts.append((list(traj.states), list(traj.actions)))
        if (process_features is not None
                and CFG.code_sim_learning_rollout_truncate_settled
                and rollouts):
            truncated = [
                truncate_settled_tail(r, process_features) for r in rollouts
            ]
            logger.info(
                "Rollout sysID: settled-tail truncation %s (tol=%g, "
                "margin=%d).", ", ".join(f"{len(r[1])}->{len(t[1])}"
                                         for r, t in zip(rollouts, truncated)),
                CFG.code_sim_learning_rollout_settle_tol,
                CFG.code_sim_learning_rollout_settle_margin)
            rollouts = truncated
        if (process_features is not None
                and CFG.code_sim_learning_rollout_segment_on_rest
                and rollouts):
            # Multiple shooting: re-anchor at observed rest points so
            # chaotic divergence cannot compound across manipulation
            # phases, and trimming can drop a chaotic phase without
            # discarding the clean cascade next to it.
            segments: List[RolloutTrajectory] = []
            for r in rollouts:
                segments.extend(split_at_rest_points(r, process_features))
            logger.info(
                "Rollout sysID: rest-point segmentation %d trajectories -> "
                "%d segments (lengths %s).", len(rollouts), len(segments),
                [len(a) for _s, a in segments])
            if segments:
                rollouts = segments
            else:
                logger.warning(
                    "Rollout sysID: segmentation found no scored motion; "
                    "keeping the whole trajectories.")
        return rollouts

    def _apply_identified_physical_params(
            self, identified: Dict[str, float]) -> None:
        """Publish identified physical params into the planning base env.

        The applied set exactly mirrors ``identified``: params applied
        by an earlier fit but absent here (e.g. dropped from a later
        artifact's PHYSICAL_PARAMS) are reverted to the env's registry
        defaults, because the env-side override is sticky per param and
        a stale value from a superseded fit would otherwise silently
        keep steering the planner. The override survives resets but not
        env recreation; ``_recreate_base_env`` re-applies from
        ``self._identified_physical_params``.
        """
        stale = set(self._identified_physical_params) - set(identified)
        if stale:
            info = self._base_env.get_physical_param_info()
            reverts = {
                name: float(info[name]["default"])
                for name in sorted(stale) if name in info
            }
            if reverts:
                self._base_env.apply_physical_param_overrides(reverts)
                logger.info(
                    "Reverted physical params dropped from the current "
                    "declaration to env defaults: %s",
                    {k: f"{v:.4f}"
                     for k, v in reverts.items()})
        self._identified_physical_params = dict(identified)
        self._base_env.apply_physical_param_overrides(identified)
        logger.info("Applied identified physical params to base env: %s",
                    {k: f"{v:.4f}"
                     for k, v in identified.items()})

    def _fit_parameters_joint_rollout(
        self,
        rules: List,
        rule_specs: List[ParamSpec],
        process_features: Dict[str, List[str]],
        num_steps: Optional[int] = None,
    ) -> Tuple[FitResult, float]:
        """Joint physical+rule fit against free-running base-sim rollouts.

        Reached when the artifact declares ``PHYSICAL_PARAMS``. Consumes
        the RAW observed trajectories rather than ``base_pred_triples``:
        physical parameters only manifest when momentum free-runs, which
        the teacher-forced triples destroy (``State`` has no
        velocities). One theta = physical + rule params, one MCMC, so
        rules cannot silently absorb physics error; with no rules this
        degenerates to pure identification. The identified physical
        values are applied in place to the planning base env, and the
        per-parameter identifiability report (posterior contraction) is
        logged so null parameters are visible rather than silently
        trusted.
        """
        physical_specs = self._physical_param_specs
        physical_names = [s.name for s in physical_specs]
        # Factory, not an instance: every rollout runs in a fresh env.
        fit_env = self._get_rollout_fit_env()
        rollouts = self._rollout_fit_trajectories(process_features)
        init_params = {
            s.name: s.init_value
            for s in physical_specs + rule_specs
        }
        anchors = physical_param_anchors(self._base_env, physical_specs)
        if not rollouts:
            logger.warning(
                "No complete trajectories for rollout sysID; keeping the "
                "declared physical-param inits unfitted.")
            result = fit_params_rollout(fit_env, [],
                                        physical_specs,
                                        process_features,
                                        rules=rules,
                                        rule_specs=rule_specs,
                                        latent_init=self._latent_init,
                                        num_steps=0,
                                        anchors=anchors)
            self._apply_identified_physical_params(
                {n: init_params[n]
                 for n in physical_names})
            return result, float("nan")

        # One scaling object per fit: every SSE/RMS below must share it
        # or their values are incomparable.
        scaling = compute_residual_scaling(rollouts, process_features)
        pre_sse = compute_rollout_sse(fit_env, rollouts, init_params,
                                      process_features, physical_names, rules,
                                      self._latent_init, scaling)
        logger.info("Rollout sysID - pre-SSE: %.6f", pre_sse)
        result, survivors, rms = fit_params_rollout_trimmed(
            fit_env,
            rollouts,
            physical_specs,
            process_features,
            rules=rules,
            rule_specs=rule_specs,
            latent_init=self._latent_init,
            num_steps=num_steps,
            scaling=scaling,
            anchors=anchors,
            rms_cache=self._explainability_cache)
        # Post-SSE and the identifiability probe are computed on the
        # SURVIVING trajectories: a trimmed (unexplainable) recording
        # would otherwise re-poison the verdicts with its noise.
        if not survivors:
            # NO fit ran (result is pinned at the declared inits).
            # Apply nothing: the planner keeps its standing belief -
            # the previous cycle's applied values - rather than being
            # reverted to baselines (or moved to this call's declared
            # inits) by data that supports neither.
            logger.warning(
                "Rollout sysID: no explainable segments this cycle; "
                "leaving the planner's physical params untouched.")
            self._record_sysid_diagnostics({}, physical_names, 0,
                                           len(rollouts), rms)
            return result, float("nan")
        probe_rollouts = survivors
        fitted = result.point_estimate
        post_sse = compute_rollout_sse(fit_env, probe_rollouts, fitted,
                                       process_features, physical_names, rules,
                                       self._latent_init, scaling)
        logger.info("Rollout sysID - post-SSE: %.6f", post_sse)

        def rollout_sse_fn(params: Dict[str, float]) -> float:
            return compute_rollout_sse(fit_env, probe_rollouts, params,
                                       process_features, physical_names, rules,
                                       self._latent_init, scaling)

        report = identifiability_report(result,
                                        rollout_sse_fn,
                                        list(physical_specs) +
                                        list(rule_specs),
                                        num_explainable=len(survivors))
        self._check_cross_cycle_consistency(result, report, physical_names)
        logger.info("Identifiability (posterior/prior contraction):\n%s",
                    format_identifiability(report))
        log_param_changes(init_params, fitted)
        self._apply_identified_physical_params(
            select_trustworthy_params(fitted, init_params, physical_names,
                                      report, anchors))
        self._record_sysid_diagnostics(report, physical_names, len(survivors),
                                       len(rollouts), rms)
        return result, post_sse

    def _check_cross_cycle_consistency(self, result: FitResult,
                                       report: Dict[str, Dict[str, Any]],
                                       physical_names: Sequence[str]) -> None:
        """Flag params whose confident MAP jumped since the previous cycle.

        The curvature probe measures local *precision*: a biased
        objective yields precisely-wrong values that the probe still
        stamps "identified" (observed: per-cycle friction fits 0.0585,
        0.0614, 0.0919, 0.0794, each with posterior_std ~0.003 -
        mutually incompatible by many sigmas). Comparing successive
        final fits in FIT space (log for log-scale params) catches
        exactly this: a jump above
        ``CFG.code_sim_learning_rollout_cross_cycle_sigma`` combined
        sigmas downgrades "identified" to weakly identified with an
        explicit note (the value is still applied - the new fit has
        strictly more data - but the planner-facing confidence is
        honest, and the explorer diagnostics pick it up). History
        records only these final per-cycle fits, not the agent's
        in-session tool fits, whose param sets churn.
        """
        k = CFG.code_sim_learning_rollout_cross_cycle_sigma
        fitted = result.point_estimate
        scales = result.scales or ["linear"] * len(result.names)
        for i, name in enumerate(result.names):
            if name not in physical_names:
                continue
            post = float(
                report.get(name, {}).get("posterior_std", float("nan")))
            scale = scales[i]
            value = fitted[name]
            prev = self._sysid_fit_history.get(name)
            if (k > 0 and prev is not None and np.isfinite(post)
                    and np.isfinite(prev[1])):
                prev_val, prev_std, prev_scale = prev
                if prev_scale == scale:
                    if scale == "log":
                        dist = abs(
                            np.log(max(value, 1e-300)) -
                            np.log(max(prev_val, 1e-300)))
                    else:
                        dist = abs(value - prev_val)
                    combined = float(np.sqrt(post**2 + prev_std**2))
                    if combined > 0 and dist / combined > k:
                        n_sigma = dist / combined
                        logger.warning(
                            "Rollout sysID cross-cycle consistency: %s "
                            "moved %.4f -> %.4f (%.1f combined sigmas > "
                            "%g) since the previous cycle; the posterior "
                            "is overconfident.", name, prev_val, value,
                            n_sigma, k)
                        entry = report.get(name)
                        if entry is not None and entry["verdict"].startswith(
                                "identified"):
                            entry["verdict"] = (
                                "weakly identified (INCONSISTENT across "
                                f"cycles: {prev_val:.4f} -> {value:.4f} is "
                                f"{n_sigma:.1f} combined sigmas)")
            self._sysid_fit_history[name] = (value, post, scale)

    def _record_sysid_diagnostics(self, report: Dict[str, Dict[str, Any]],
                                  physical_names: Sequence[str],
                                  num_survivors: int, num_segments: int,
                                  rms: List[float]) -> None:
        """Digest the fit's weak spots for the next explore phase.

        Generic (domain-free) statements of what the data could not
        support - unexplainable segments, parameters the rollouts do not
        constrain, cross-cycle conflicts - phrased as experiment
        objectives. The explorer appends this to its guidance so the
        agent designs interactions that fill the gaps, instead of
        relying on whatever manipulation data the tasks happen to
        produce.
        """
        lines: List[str] = []
        dropped = num_segments - num_survivors
        if dropped > 0:
            lines.append(
                f"- {dropped} of {num_segments} recorded motion segments "
                "were unexplainable at ANY physical parameters (best RMS "
                f"{[f'{r:.3g}' for r in rms]}): their dynamics are not "
                "repeatable under replay. Prefer experiments whose outcome "
                "is dominated by object dynamics rather than prolonged "
                "robot-object contact: actuate cleanly, then let the scene "
                "evolve and settle on its own.")
        for name in physical_names:
            entry = report.get(name, {})
            verdict = entry.get("verdict", "unknown")
            if verdict.startswith("identified"):
                if entry.get("flat_wide"):
                    interval = entry["flat_interval"]
                    lines.append(
                        f"- physical param '{name}': the data cannot "
                        f"distinguish values in [{interval[0]:.4g}, "
                        f"{interval[1]:.4g}] (the applied value is the "
                        "interval edge nearest the baseline belief). An "
                        "experiment whose observable outcome DIFFERS across "
                        "this interval would pin it down.")
                continue
            lines.append(
                f"- physical param '{name}': {verdict}. An experiment whose "
                "observable outcome CHANGES when this parameter changes "
                "would identify it; if none exists, drop it from "
                "PHYSICAL_PARAMS.")
        self._last_sysid_diagnostics = ("\n".join(lines) if lines else "")

    def _sync_tool_context(self) -> None:
        super()._sync_tool_context()
        self._tool_context.sysid_diagnostics = (self._last_sysid_diagnostics
                                                or None)

    # ── Partial-observability (latent) support ───────────────────
    # Reached only when the loaded rules use the recurrent 5-arg
    # signature (``has_latent_rules``). Legacy 3-arg simulators never
    # enter these paths, so fully-observable behavior is unchanged.

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

    def _fit_parameters_recurrent(
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        process_features: Dict[str, List[str]],
        num_steps: Optional[int] = None,
    ) -> Tuple[FitResult, float]:
        """MCMC over the recurrent (per-trajectory) SSE.

        Counterpart to :func:`fitting.fit_rule_parameters` for rules
        that carry a latent block. Re-groups the flat
        ``base_pred_triples`` into per-trajectory chunks (latent threads
        within a trajectory, not across) via the lengths cached in
        ``self._fit_trajectories``; falls back to a single trajectory if
        no grouping info exists. Delegates the actual fit/log to
        :func:`fitting.fit_rule_parameters_latent` so the agent's
        ``sim.fit`` surface scores latent rules through the exact
        same path.
        """
        groups = self._group_triples_by_trajectory(base_pred_triples)
        if not groups:
            logger.warning("No trajectory groups for recurrent fitting; "
                           "falling back to single-trajectory rollout.")
            groups = [base_pred_triples]
        return fit_rule_parameters_latent(rules,
                                          specs,
                                          groups,
                                          self._latent_init,
                                          process_features,
                                          num_steps=num_steps)

    def _oracle_param_sse_recurrent(
        self,
        rules: List,
        base_pred_triples: List[Tuple[State, Action, State]],
        process_features: Dict[str, List[str]],
        noise_sigma: float,
    ) -> float:
        """Oracle-param SSE via the recurrent (latent-threaded) rollout.

        Latent counterpart to :meth:`_oracle_param_sse`'s per-transition
        body. The per-feature ``log_sse_breakdown`` is per-transition
        and so omitted; the recurrent rollout already reports its SSE.
        """
        groups = self._group_triples_by_trajectory(base_pred_triples)
        if not groups:
            logger.warning("No trajectory groups for recurrent oracle SSE; "
                           "falling back to single-trajectory rollout.")
            groups = [base_pred_triples]
        sse = compute_sse_recurrent(rules, groups, self._fitted_params,
                                    self._latent_init, process_features)
        fit_ll = -0.5 * sse / (noise_sigma**2)
        logger.info(
            "Oracle params (recurrent) - SSE: %.6f  log-likelihood: %.2f", sse,
            fit_ll)
        for name, val in sorted(self._fitted_params.items()):
            logger.info("  %-30s  %.4f", name, val)
        return sse

    def _attach_initial_latent(self, task: Task) -> Task:
        """Seed ``task.init.latent`` with the initial latent block.

        Refinement starts at ``task.init`` (the planner's ``traj[0]``), so
        the combined simulator must find a well-formed latent there. If no
        ``LATENT_INIT`` was loaded (or the resulting block is empty), leave
        the task alone so downstream code keeps the legacy
        ``state.latent is None`` behaviour. Overrides the no-op default in
        :class:`AgentModelBasedApproach`.
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

    def materialise_latent(
        self,
        traj: LowLevelTrajectory,
    ) -> List[Optional[Dict[str, Any]]]:
        """Roll a trajectory through the rules; return per-step latent.

        Used by :func:`evaluate_predicate_quality` so latent-aware
        predicates can be scored against meaningful latent values.
        Returned list aligns with ``traj.states``; entry ``i`` is the
        latent *before* predicates are evaluated at state ``i``. If no
        rules are loaded, every entry is ``None`` so latent-aware
        classifiers fall back to their default branch.
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
                # If a rule crashes, fall back to None for the remaining
                # steps so predicate evaluation continues.
                out.extend([None] * (len(traj.states) - len(out)))
                return out
            out.append(dict(latent))
        return out

    def _build_latent_combined_simulator(
            self) -> Callable[[State, Action], State]:
        """Compose base env + recurrent rules; carry latent on state.latent.

        The latent block rides on the opaque ``State.latent`` field, so
        backtracking restores it per search node. The simulator reads
        ``state.latent`` on entry, threads it through the rules, and
        attaches the updated latent to the returned state. If
        ``state.latent`` is None (e.g. the very first state), falls back to
        ``init_latent``. The latent-free ``learned_simulator`` used by
        :meth:`_build_combined_simulator` is bypassed.
        """
        assert self._process_rules is not None, (
            "_build_latent_combined_simulator called before rules loaded")
        rules: List = self._process_rules
        latent_init = self._latent_init
        # Reference the dict (not its values) so MCMC param updates are
        # picked up by the closure live.
        params = self._fitted_params

        def combined_simulate(state: State, action: Action) -> State:
            # `state` is one sample of the augmented state: observable
            # features in `.data` + inferred latent dims in `.latent`.
            # Deep-copy the incoming latent so this call can't mutate the
            # caller's state and sibling branches at the same parent stay
            # independent. The latent nests a per-jug dict, so a shallow
            # ``dict(...)`` would still alias (and clobber) it.
            latent = (copy.deepcopy(state.latent) if state.latent is not None
                      else init_latent(latent_init, params))
            try:
                base_state = self._base_env.simulate(state, action)
            except pybullet.error as e:
                logging.warning(
                    "PyBullet error in recurrent combined_simulate (%s); "
                    "recreating base env and retrying.", e)
                self._recreate_base_env()
                base_state = self._base_env.simulate(state, action)
            # Repair features the backtracking reset couldn't round-trip
            # (e.g. bubbling_level derived from a hidden heat_level): the
            # base env's value is meaningless there, so restore the carried
            # value before the rules read it.
            self._restore_unreconstructible_process_features(base_state, state)
            # Single-step history window; rules needing longer context
            # must accumulate it in ``latent``.
            history: List[Tuple[State,
                                Optional[Action]]] = [(base_state, action)]
            updates = apply_rules_with_latent(base_state, latent, history,
                                              rules, params)
            next_state = (merge_updates(base_state, updates)
                          if updates else base_state)
            next_state.latent = latent
            return next_state

        return combined_simulate

    # ── Process-feature inference ────────────────────────────────

    @staticmethod
    def _compute_base_pred_triples(
        obs_triples: List[Tuple[State, Action, State]],
        base_env: Any,
    ) -> List[Tuple[State, Action, State]]:
        """Replace each ``s_t`` with the base sim's one-step prediction."""
        return [(base_env.simulate(s, a), a, s_next)
                for s, a, s_next in obs_triples]

    @staticmethod
    def _infer_process_features_from_residuals(
        obs_triples: List[Tuple[State, Action, State]],
        base_pred_triples: List[Tuple[State, Action, State]],
        abs_tol: float = 1e-4,
        rel_tol: float = 1e-3,
        min_hits: int = 3,
    ) -> Dict[str, List[str]]:
        """Features whose base-sim prediction diverges from observation.

        Flags ``(type, feat)`` if ``|pred - obs| > rel_tol*|obs| + abs_tol``
        on at least ``min_hits`` triples. The ``min_hits`` floor keeps
        one-off PyBullet jitter from leaking base-handled features into the set.
        """
        del obs_triples  # objects are identical across both triple lists
        pairs = [(s_base, s_obs) for s_base, _, s_obs in base_pred_triples]
        hits: Dict[Tuple[str, str], int] = {}
        for _, _, tn, feat, pred, obs in iter_feature_residuals(pairs):
            if abs(pred - obs) > rel_tol * abs(obs) + abs_tol:
                hits[(tn, feat)] = hits.get((tn, feat), 0) + 1
        out: Dict[str, List[str]] = {}
        for (t, f), n in hits.items():
            if n >= min_hits:
                out.setdefault(t, []).append(f)
        return {t: sorted(fs) for t, fs in out.items()}

    @staticmethod
    def _log_feature_set_diff(
        a: Dict[str, List[str]],
        b: Dict[str, List[str]],
        a_label: str,
        b_label: str,
    ) -> None:
        """Log set-difference between two {type: [feats]} maps."""
        a_pairs = {(t, f) for t, fs in a.items() for f in fs}
        b_pairs = {(t, f) for t, fs in b.items() for f in fs}
        only_a = sorted(a_pairs - b_pairs)
        only_b = sorted(b_pairs - a_pairs)
        common = a_pairs & b_pairs
        logger.info(
            "Feature-set diff: %s vs %s (%d common, %d only-%s, %d only-%s)",
            a_label, b_label, len(common), len(only_a), a_label, len(only_b),
            b_label)
        if only_a:
            logger.info("  only in %s: %s", a_label, only_a)
        if only_b:
            logger.info("  only in %s: %s", b_label, only_b)

    @staticmethod
    def _format_predicate_signatures(predicates: Set[Predicate]) -> str:
        """Pretty-print predicates as ``Name(type1, type2)`` lines.

        Mirrors the ``## Available Predicates`` block in
        ``bilevel_sketch.build_solve_prompt``.
        """
        lines = []
        for pred in sorted(predicates, key=lambda p: p.name):
            type_sig = ", ".join(t.name for t in pred.types)
            line = f"  {pred.name}({type_sig})"
            if pred.natural_language_assertion is not None:
                names = [t.name for t in pred.types]
                line += f" - {pred.natural_language_assertion(names)}"
            lines.append(line)
        return "\n".join(lines)

    def _make_evaluate_trajectory_fn(self) -> Any:
        """Build the ``evaluate_trajectory`` helper exposed in the synthesis
        exec namespace (next to ``is_goal_state``).

        The returned function scores a concrete state sequence with the
        task's env-defined ``TaskEvaluator`` and returns only the public
        pair (dict of reward/solved) - never the evaluator itself, and
        never the certificate's internal legitimacy verdict or reason
        (the agent infers the scoring rules from the stated objective
        and the outcomes it observes; goal-atom termination it can
        check itself via ``is_goal_state``). ``actions`` may be
        ``Action`` objects (labeled via their producing options),
        pre-built ``(option_name, object_names[, params])`` labels, or
        ``None`` (kinematics-only scoring).
        """
        tasks = self._train_tasks

        def evaluate_trajectory(states: Sequence[State],
                                actions: Optional[Sequence[Any]] = None,
                                task_idx: int = 0) -> Dict[str, Any]:
            if not 0 <= task_idx < len(tasks):
                raise ValueError(f"task_idx {task_idx} out of range "
                                 f"(0-{len(tasks) - 1}).")
            evaluator = tasks[task_idx].evaluator
            if evaluator is None:
                raise ValueError(
                    f"Train task {task_idx} defines no task evaluator.")
            if not states:
                raise ValueError("`states` must be a non-empty sequence.")
            step_options: Optional[Sequence[Any]] = None
            if actions is not None:
                acts = list(actions)
                if acts and isinstance(acts[0], Action):
                    step_options = step_option_labels(acts)
                else:
                    step_options = acts
            verdict = evaluate_states_with(evaluator,
                                           list(states),
                                           step_options,
                                           sim_env=getattr(
                                               self._option_model, "sim_env",
                                               None))
            return {
                "reward": verdict["reward"],
                "solved": verdict["solved"],
            }

        return evaluate_trajectory

    @staticmethod
    def _format_trajectory_listing(
            trajectories: List[LowLevelTrajectory]) -> str:
        """Render a per-trajectory listing with provenance tags.

        Each interaction trajectory shows the simulator / predicates
        snapshot used to generate the plan that collected it (if
        tracked). Demo trajectories list as ``demo``. Listed in the same
        order the agent sees them via the ``trajectories`` var.
        """
        if not trajectories:
            return ""
        lines = ["Trajectory roster (matches the `trajectories` list):"]
        for idx, traj in enumerate(trajectories):
            kind = "demo" if traj.is_demo else "interaction"
            try:
                task_str = f"task {traj.train_task_idx}"
            except AssertionError:
                task_str = "task ?"
            provenance: List[str] = []
            sim_v = traj.source_simulator_version
            preds_v = traj.source_predicates_version
            if sim_v:
                provenance.append(f"sim {sim_v}")
            if preds_v:
                provenance.append(f"predicates {preds_v}")
            tail = (f" - generated using {', '.join(provenance)}"
                    if provenance else "")
            if traj.env_reward is not None:
                solved = int(
                    bool(traj.env_terminated) and not traj.env_rejected)
                tail += (f" - env reward={traj.env_reward:.2f} "
                         f"(solved={solved})")
            lines.append(f"  [{idx}] {kind}, {task_str}{tail}")
        return "\n".join(lines) + "\n"

    def _format_objective_block(self) -> str:
        """The env's public task objective (reward form), or empty.

        Emitted when a train task's evaluator states an objective. The
        statement is public by design: it contains the reward FORM
        (success condition + costs), never oracle quantities like the
        true minimum block count.
        """
        description = next(
            (t.evaluator.objective_description()
             for t in self._train_tasks if t.evaluator is not None
             and t.evaluator.objective_description()), "")
        if not description:
            return ""
        return f"""\
## Task objective (env ground-truth reward)
{description}

The trajectory roster above shows each interaction episode's \
env-computed reward. In `run_python`, \
`evaluate_trajectory(states, actions=None, task_idx=0)` scores any \
state sequence with the same ground-truth evaluator - a collected \
trajectory's `states`/`actions`, or a rollout of YOUR simulator \
(there the verdict is only as trustworthy as your simulator). It \
returns {{reward, solved}}. `solved` means the episode is scored as \
a success; a rollout can reach the goal atoms and still be \
solved=False.

"""

    def _format_prior_state_block(self, base: str) -> str:
        """Tell the agent about any simulator/predicates left over from a
        previous learning cycle.

        Returns a paragraph the agent can act on (read the files first
        and treat this cycle as incremental refinement) or an empty
        string if no prior state exists. The base sandbox dir is scanned
        for ``simulator.py`` / ``predicates.py``.
        """
        prior: List[str] = []
        sim_path = os.path.join(base, "simulator.py")
        preds_path = os.path.join(base, "predicates.py")
        if os.path.isfile(sim_path):
            prior.append("`./simulator.py`")
        if os.path.isfile(preds_path):
            prior.append("`./predicates.py`")
        if not prior:
            return ""
        joined = " and ".join(prior)
        return f"""\
Prior cycle state: {joined} already exist in the sandbox from a previous \
learning cycle. Read them first - they are the previous cycle's committed \
result and a reasonable starting point for incremental refinement (though \
a fresh rewrite is fine if the prior approach looks fundamentally wrong). \
Earlier versions are in `./simulator_versions/` and \
`./predicates_versions/` (named `cycle_XXX_vers_YYY_*.py`); \
cross-reference the trajectory roster's provenance tags against those \
files to see exactly which rules and predicates produced each failed plan.

"""

    @staticmethod
    def _load_simulator_from_module_file(
        path: str,
        trajectories: Optional[List[LowLevelTrajectory]] = None,
    ) -> Tuple[Optional[List], Optional[List[ParamSpec]], Optional[Dict[
            str, List[str]]], Optional[Dict[str, Any]]]:
        """Load PROCESS_RULES, PARAM_SPECS, PROCESS_FEATURES from one file.

        Execs ``path`` once in a fresh namespace and returns ``(rules,
        specs, features, ns)``, where ``ns`` is that exec namespace so
        callers/subclasses can read extra exports (e.g. ``LATENT_INIT``)
        without re-execing. ``ns`` is ``None`` only when no exec
        happened (missing file or exec failure). ``rules``/``specs`` are
        ``None`` when ``PROCESS_RULES``/``PARAM_SPECS`` is absent (the
        caller treats that as failure); ``features`` may be ``None``
        independently (``PROCESS_FEATURES`` is then asserted by the
        caller).
        """
        if not os.path.isfile(path):
            logger.warning("No simulator file at %s.", path)
            return None, None, None, None

        ns: Dict[str, Any] = {
            "np": np,
            "ParamSpec": ParamSpec,
            "trajectories": trajectories or [],
        }
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            exec(code, ns)  # pylint: disable=exec-used
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to exec %s.", path, exc_info=True)
            return None, None, None, None

        rules, specs, features = read_simulator_components(ns)
        # A physics-only artifact (PHYSICAL_PARAMS with no process rules)
        # is valid: the base sim carries all the dynamics once its
        # parameters are identified, so rules/specs default to empty.
        physics_only = read_physical_param_specs(ns) is not None
        if rules is None:
            if not physics_only:
                logger.warning("Simulator file %s missing PROCESS_RULES.",
                               path)
                return None, None, None, ns
            rules = []
        if specs is None:
            if not physics_only:
                logger.warning("Simulator file %s missing PARAM_SPECS.", path)
                return None, None, None, ns
            specs = []

        logger.info("Loaded %d rules, %d param specs from %s%s.", len(rules),
                    len(specs), path,
                    " (physics-only artifact)" if physics_only else "")
        return rules, specs, features, ns

    # ── Static helpers ───────────────────────────────────────────

    def _write_structs_reference(self) -> str:
        """Write key struct sources to the sandbox; return the agent-visible
        path."""
        # pylint: disable=import-outside-toplevel,reimported
        from predicators.structs import Action as _Action
        from predicators.structs import LowLevelTrajectory as _LLT
        from predicators.structs import Object as _Object
        from predicators.structs import State as _State
        from predicators.structs import Type as _Type

        source = "\n\n".join(
            inspect.getsource(cls)
            for cls in [_Type, _Object, _State, _Action, _LLT])

        base = self._tool_context.sandbox_dir or self._get_log_dir()
        ref_dir = os.path.join(base, "reference")
        os.makedirs(ref_dir, exist_ok=True)
        ref_path = os.path.join(ref_dir, "structs.py")
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(source)

        # Same backend-dependent agent-visible path mapping as
        # _resolve_synthesis_paths.
        if CFG.agent_sdk_use_local_sandbox:
            return "./reference/structs.py"
        if self._tool_context.sandbox_dir:
            return "/sandbox/reference/structs.py"
        return ref_path

    @staticmethod
    def _extract_obs_triples(
        trajectories: List[LowLevelTrajectory],
    ) -> List[Tuple[State, Action, State]]:
        """Extract observed (s_t, action_t, s_{t+1}) triples."""
        triples: List[Tuple[State, Action, State]] = []
        for traj in trajectories:
            for i in range(len(traj.actions)):
                triples.append(
                    (traj.states[i], traj.actions[i], traj.states[i + 1]))
        return triples

    def _recreate_base_env(self) -> None:
        """Reconnect after a PyBullet physics-server crash."""
        try:
            client_id = self._base_env._physics_client_id  # type: ignore[attr-defined]  # pylint: disable=protected-access
            pybullet.disconnect(client_id)
        except Exception:  # pylint: disable=broad-except  # client may already be dead
            pass
        logging.warning(
            "PyBullet physics client crashed; recreating base env "
            "(use_gui=%s).", CFG.option_model_use_gui)
        self._base_env = create_new_env(CFG.env,
                                        do_cache=False,
                                        use_gui=CFG.option_model_use_gui,
                                        skip_process_dynamics=True)
        # A fresh env comes up with built-in physics; re-assert any
        # identified physical params (the in-place override does not
        # survive env recreation).
        if self._identified_physical_params:
            self._base_env.apply_physical_param_overrides(
                self._identified_physical_params)
        # The option model's transient certificate env rides on
        # _base_env; re-point it so probes don't run against the dead
        # client's stale physics overrides.
        if self._option_model is not None and \
                getattr(self._option_model, "sim_env", None) is not None:
            self._option_model.sim_env = self._base_env

    @contextmanager
    def _fresh_validation_env_scope(self) -> Iterator[None]:
        """Run the option model on a freshly constructed base env.

        Installed as ``ToolContext.validation_env_scope`` so
        ``evaluate_option_plan``'s capture-validation rollouts each sample
        a fresh physics world. The shared ``_base_env``'s reset cannot
        reconstruct state exactly (solver warm-start state, velocity
        residuals, near-matching bodies skipped by the reconstruction diff
        - the same mechanism measured in :func:`rollout_states`), so
        repeats on it are correlated with each other and systematically
        offset from the fresh env the real episode runs in
        (run_20260717_182321: a placement swept 20/20 on the shared env
        validated 3/3, then missed the target on the real rollout).

        Swaps ``_base_env`` (the learned combined simulator reads it
        dynamically), the option model's ``sim_env`` (backs certificate
        probes), and - for the pre-learning model, whose simulator is the
        bound method ``_base_env.simulate`` - the model's ``_simulator``.
        Everything is restored and the fresh env disposed on exit,
        including the replacement env a mid-rollout PyBullet-crash
        recovery (``_recreate_base_env``) may have installed.
        """
        fresh = create_new_env(CFG.env,
                               do_cache=False,
                               use_gui=False,
                               skip_process_dynamics=True)
        if self._identified_physical_params:
            fresh.apply_physical_param_overrides(
                self._identified_physical_params)
        prev_env = self._base_env
        # Typed Any: sim_env and _simulator are dynamic attributes not on
        # _OptionModelBase.
        model: Any = self._option_model
        prev_sim = getattr(model, "_simulator", None)
        rebind_sim = getattr(prev_sim, "__self__", None) is prev_env
        prev_sim_env = getattr(model, "sim_env", None)
        self._base_env = fresh
        if rebind_sim:
            model._simulator = fresh.simulate  # pylint: disable=protected-access
        if prev_sim_env is not None:
            model.sim_env = fresh
        try:
            yield
        finally:
            current = self._base_env
            self._base_env = prev_env
            if rebind_sim:
                model._simulator = prev_sim  # pylint: disable=protected-access
            if prev_sim_env is not None:
                model.sim_env = prev_sim_env
            if current is not prev_env:
                try:
                    dispose_env(current)
                except Exception:  # pylint: disable=broad-except
                    pass  # client already dead (crashed mid-rollout)

    def _restore_unreconstructible_process_features(self, base_state: State,
                                                    prev_state: State) -> None:
        """Restore process features the base env's reset couldn't round-trip.

        When the option model backtracks (jumps to a non-current node), the
        base PyBullet env reconstructs the State from observables only, so a
        feature derived from a hidden sim-feature (e.g. ``bubbling_level``,
        projected from a hidden ``heat_level``) comes back at its default
        (0) instead of its carried value. The learned model *owns* those
        features, so the base value is meaningless; overwrite ``base_state``
        with the value carried in ``prev_state`` before the rules read it.

        Scoping is the key to not breaking co-owned features: restore only
        the intersection of (a) the env's reported unreconstructible set for
        this step and (b) the declared ``PROCESS_FEATURES``. A kinematic,
        base-reconstructible feature that a robot legitimately moves (e.g. a
        wind-blown ball's ``x, y`` in the fans env) round-trips through the
        reset, so it never enters the env's set and is left to the base sim.
        On sequential rollouts the env's set is empty, so this is a no-op.
        """
        lossy = getattr(self._base_env, "_last_unreconstructible_features",
                        None)
        if not lossy or not self._process_features:
            return
        for obj, feat in lossy:
            if feat in self._process_features.get(obj.type.name, []) \
                    and obj in prev_state.data:
                base_state.set(obj, feat, prev_state.get(obj, feat))

    def _build_combined_simulator(
        self,
        learned_simulator: LearnedSimulator,
    ) -> Callable[[State, Action], State]:
        """Compose base env with learned step-level dynamics.

        Captures ``self`` so the closure can recreate ``_base_env`` and
        retry once on a PyBullet crash (common on macOS Metal + GUI).
        When the loaded rules carry a latent block (partial
        observability), delegates to
        :meth:`_build_latent_combined_simulator`, which threads
        ``state.latent`` through the recurrent rules instead of the
        latent-free ``learned_simulator``.
        """
        if has_latent_rules(self._process_rules or []):
            return self._build_latent_combined_simulator()

        def combined_simulate(state: State, action: Action) -> State:
            try:
                base_state = self._base_env.simulate(state, action)
            except pybullet.error as e:
                logging.warning(
                    "PyBullet error in combined_simulate (%s); "
                    "recreating base env and retrying.", e)
                self._recreate_base_env()
                base_state = self._base_env.simulate(state, action)
            self._restore_unreconstructible_process_features(base_state, state)
            updates = learned_simulator.predict_step(base_state)
            if not updates:
                return base_state
            return merge_updates(base_state, updates)

        return combined_simulate

    def _build_synthesis_system_prompt(self) -> str:
        """Render the synthesis system prompt from the module template.

        Substitutes the placeholders that vary per instance: the
        rule-signature blocks (the partial-observability subclass
        overrides the two signature hooks so its prompt presents only
        the recurrent 5-arg form as canonical - the 3-arg form sitting
        beside the PO guidance previously led the agent to write a
        3-arg rule the recurrent engine rejects), the optional
        PHYSICAL_PARAMS section (env parameter menu), the
        scene-visualization hint (tool surface), and the subclass extra
        section.
        """
        prompt = _SYNTHESIS_SYSTEM_PROMPT_TEMPLATE
        prompt = prompt.replace("__RULE_SIGNATURE_SECTION__",
                                self._rule_signature_section())
        prompt = prompt.replace("__PROCESS_RULE_SIGNATURE__",
                                self._process_rule_signature())
        prompt = prompt.replace("__PHYSICAL_PARAMS_SECTION__",
                                self._physical_params_prompt_section())
        prompt = prompt.replace("__SCENE_VIZ_HINT__", self._scene_viz_hint())
        extra = self._extra_synthesis_system_prompt()
        extra_block = "\n" + extra.rstrip() + "\n" if extra else ""
        return prompt.replace("__SYNTHESIS_PROMPT_EXTRA__", extra_block)

    @staticmethod
    def _scene_viz_hint() -> str:
        """The find-the-anchor-offset sentence.

        The probe is unconditional in synthesis sessions, so the hint
        always names its staging + overlay surface.
        """
        return ("use the `sim` probe in `run_python`: "
                "`sim.reset(task_idx=..., "
                "mods={...})` to stage a representative state from each "
                "bucket and `sim.render(label, annotations=[...])` to "
                "overlay, on one render, the recorded origin and the "
                "positions where the effect did vs. did not fire")

    def _physical_params_prompt_section(self) -> str:
        """Markdown for the optional PHYSICAL_PARAMS (system-ID) block.

        Built from the base env's revealed parameter menu
        (``get_physical_param_info``); empty when the env reveals none,
        so non-parameterized envs never see the feature mentioned.
        """
        info: Dict[str, Dict[str, Any]] = {}
        # getattr chain (not plain attribute access) so the prompt still
        # renders on instances without a base env (see the bare-instance
        # rendering tests in test_agent_sim_prompt_formatting.py).
        base_env = getattr(self, "_base_env", None)
        getter = getattr(base_env, "get_physical_param_info", None)
        if callable(getter):
            info = getter() or {}
        if not info:
            return ""
        lines = [
            "",
            "## Optional: base-sim system identification "
            "(`PHYSICAL_PARAMS`)",
            "",
            "The base sim's rigid-body physics is itself parameterized, "
            "and its built-in values may be MIS-CALIBRATED (the real "
            "environment may run different physics). It reveals these "
            "tunable parameters:",
            "",
        ]
        for name, meta in info.items():
            scale_note = (", fitted in log-space"
                          if meta.get("scale") == "log" else "")
            lines.append(f"- `{name}` (built-in {meta['default']:.4g}, fit "
                         f"box [{meta['lo']:.4g}, {meta['hi']:.4g}]"
                         f"{scale_note}): {meta['description']}")
        lines.extend([
            "",
            "If observed trajectories diverge from the base sim on "
            "*rigid-body motion itself* (not a hidden process layered on "
            "top of it), declare a fourth export:",
            "",
            "```python",
            "PHYSICAL_PARAMS: List[ParamSpec]  # subset of the names "
            "above; init = your hypothesis, lo/hi from the box",
            "```",
            "",
            "Guidance:",
            "",
            "- **Declare a sparse subset**, not everything: include only "
            "parameters you hypothesize are both execution-relevant and "
            "identifiable from the data (a collision parameter cannot be "
            "identified from trajectories that contain no collisions). "
            "`sim.fit()` returns a per-parameter identifiability "
            "report (posterior contraction); drop any parameter reported "
            "as NOT identified - its fitted value is arbitrary noise.",
            "- With `PHYSICAL_PARAMS` declared, the fit switches to "
            "matching **free-running rollouts** of full trajectories "
            "(momentum accrues in-sim, which the per-step teacher-forced "
            "fit destroys), and physical + rule parameters are fit "
            "**jointly** in one posterior, so rules cannot silently "
            "absorb physics error.",
            "- A physics-only artifact is valid: `PROCESS_RULES = []` and "
            "`PARAM_SPECS = []` with a non-empty `PHYSICAL_PARAMS` means "
            "the calibrated base sim carries all the dynamics. "
            "`PROCESS_FEATURES` must still be declared - it defines which "
            "features the rollout is scored on (e.g. the pose features "
            "of the objects whose motion you are calibrating).",
            "- After the fit, the identified values are applied to the "
            "planning base env, so probe rollouts and "
            "test-time planning use the calibrated physics.",
        ])
        return "\n".join(lines) + "\n"

    def _rule_signature_section(self) -> str:
        """Markdown for the '### Rule signature' block.

        Fully-observable default: the legacy 3-arg signature. The
        partial-observability subclass overrides this with the recurrent
        5-arg signature so its prompt never advertises the 3-arg form as
        canonical.
        """
        return _FO_RULE_SIGNATURE_SECTION

    def _process_rule_signature(self) -> str:
        """The ``def`` line used in the geometric-gate example.

        Matches the signature advertised by
        :meth:`_rule_signature_section` so the worked example doesn't
        contradict the canonical signature.
        """
        return "def process_rule(state, updates, params):"
