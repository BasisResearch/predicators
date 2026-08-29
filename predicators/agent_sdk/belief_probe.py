"""Exploration probe API exposed to agents via ``run_python``.

``BeliefProbe`` is a thin facade over the machinery the curated tools
already use - ``parse_sketch_from_text`` (plan grammar),
``execute_plan_forward`` (forward executor over the option model), the
tools' state-modification and rendering helpers - so probe rollouts
behave identically to ``evaluate_option_plan`` rollouts. What it adds is
composability: the agent can set the sim to any task state (or a
modified copy), read full-precision features, run partial plans, render,
snapshot/restore, and write sweep loops in one ``run_python`` call
instead of one tool round-trip per experiment.

By construction nothing the probe executes can be captured as the
answer - submission happens only through ``evaluate_option_plan`` on the
true initial state. The task evaluator is reachable, but only as a
read-only preview: ``run(trials>=2, solved=True)`` and
``refine(require_solved=True)`` score rollouts through the same gate the
capture path uses (see ``_require_solved_evaluator``).

In synthesis sessions the same facade probes the CANDIDATE simulator
(the ``simulator.py`` under edit, freshly fitted - see
``BeliefProbe._option_model``), ``fit`` exposes the fitting stack
(``ctx.probe_fit_provider``), and validation is hand-composed:
``fit()`` then ``refine`` then a continuous ``run`` of the refined
plan (the forward pass; ``run`` reports the first step whose subgoal
annotation failed to hold).
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, \
    Sequence, Tuple, Union

from predicators import utils
from predicators.agent_sdk.config import RefinementConfig, ToolSurfaceConfig, \
    ValidationConfig
from predicators.agent_sdk.parallel_rollouts import prefetch_parallel
from predicators.agent_sdk.tools.context import absolute_rollout_seed, \
    decorrelated_rollout_seed
from predicators.agent_sdk.tools.scene import apply_state_modifications, \
    draw_pybullet_annotation, render_pybullet_image, render_scene_image
from predicators.agent_sdk.tools.verdicts import _EvalStateCollector, \
    evaluate_states_with, load_ground_sampler_fns, make_solved_check
from predicators.structs import State, Task

if TYPE_CHECKING:
    from predicators.agent_sdk.tools import ToolContext

# Modification format accepted by ``reset``: either the tools' list form
# ``[{"object": name, "features": {feat: val}}]`` or the terser dict form
# ``{name: {feat: val}}``.
Modifications = Union[List[Dict[str, Any]], Dict[str, Dict[str, float]]]


class ProbeBudgetExceeded(Exception):
    """A probe call ran past a wall-clock budget.

    Raised cooperatively at probe checkpoints (every sim call) when the
    run_python per-call limit or the solve attempt's wall clock has
    expired. ``run_python`` catches it specially: the code's printed
    output so far is returned with the budget message appended, so a
    stopped sweep still hands the agent its partial results.
    """


def _check_time_budget(ctx: "ToolContext") -> None:
    """Raise :class:`ProbeBudgetExceeded` when a wall-clock budget is up.

    Never fires during the final-submission nudge
    (``ctx.capture_best_effort_plan``): with the budget spent, the one
    thing left is submitting, and blocking that would forfeit the task.
    """
    if ctx.capture_best_effort_plan:
        return
    now = time.monotonic()
    attempt_dl = ctx.attempt_deadline
    if attempt_dl is not None and now > attempt_dl:
        raise ProbeBudgetExceeded(
            "the attempt's wall-clock exploration budget is exhausted. Stop "
            "exploring NOW and submit your single best plan via "
            "evaluate_option_plan on the current task (omit task_idx).")
    call_dl = ctx.python_call_deadline
    if call_dl is not None and now > call_dl:
        call_timeout = ToolSurfaceConfig.from_cfg().python_call_timeout
        raise ProbeBudgetExceeded(
            f"this run_python call exceeded its "
            f"{call_timeout:.0f}s time limit "
            "and was stopped between sim calls; output printed so far is "
            "returned above. Large sweeps are expensive - narrow the "
            "candidate set (coarse-to-fine, fewer perturbations per "
            "candidate) and split work across calls so each returns "
            "within the limit.")


def _count_rollout(ctx: "ToolContext", n: int = 1) -> None:
    """Meter full-plan rollouts for the attempt budget footer."""
    ctx.attempt_rollout_count += n


def _fmt_params(params: Any) -> str:
    """``[p1, p2]`` rendering used everywhere the probe prints params, so
    values the agent copies out of one report parse back identically in
    another."""
    return "[" + ", ".join(f"{float(p):.4g}" for p in params) + "]"


def _fmt_option(option: Any) -> str:
    """``Name(obj1, obj2)[p1, p2]`` signature for reports."""
    objs = ", ".join(o.name for o in option.objects)
    return f"{option.name}({objs}){_fmt_params(option.params)}"


# Contact-pair lines reported per step before eliding the rest: enough
# for a busy step's story, small enough to not drown the step report.
_MAX_CONTACT_LINES_PER_STEP = 12


def _attach_step_contacts(events: List[Dict[str, Any]],
                          step_dicts: List[Dict[str, Any]]) -> None:
    """Bucket recorded contact events into per-option-step summaries.

    ``events`` come from ``stop_contact_recording`` (step-ordered, one
    env step per low-level action, so cumulative ``num_actions`` gives
    the option-step boundaries). Each step dict gains ``contacts``: one
    line per contact pair with its 0-based action span within the step,
    robot links listed first. The LAST step's bucket absorbs any events
    past the counted boundary: a partially-executed failing option
    reports ``num_actions=0`` while the env genuinely stepped, and
    dropping those events would blank the contacts of exactly the step
    under diagnosis.
    """
    idx = 0
    start = 0
    for i, s in enumerate(step_dicts):
        is_last = i == len(step_dicts) - 1
        end = float("inf") if is_last else start + s["num_actions"]
        # Insertion-ordered dict doubles as the report order.
        spans: Dict[Tuple[str, str], Tuple[int, int]] = {}
        while idx < len(events) and events[idx]["step"] <= end:
            ev = events[idx]
            idx += 1
            if ev["step"] <= start:
                continue
            a, b = ev["a"], ev["b"]
            # Canonicalize the unordered pair: robot link first, else
            # lexicographic, so the two orientations merge into one span.
            if a.startswith("robot:") != b.startswith("robot:"):
                if b.startswith("robot:"):
                    a, b = b, a
            elif a > b:
                a, b = b, a
            key = (a, b)
            rel = ev["step"] - start - 1
            if key not in spans:
                spans[key] = (rel, rel)
            else:
                spans[key] = (spans[key][0], rel)
        lines = []
        for key, (lo, hi) in spans.items():
            span = f"action {lo}" if lo == hi else f"actions {lo}-{hi}"
            lines.append(f"{key[0]} <-> {key[1]} ({span})")
        if len(lines) > _MAX_CONTACT_LINES_PER_STEP:
            extra = len(lines) - _MAX_CONTACT_LINES_PER_STEP
            lines = lines[:_MAX_CONTACT_LINES_PER_STEP] + [
                f"... +{extra} more pairs"
            ]
        s["contacts"] = lines
        start += s["num_actions"]


class _StrLikeResult:
    """String conveniences shared by the probe result types.

    The results print like strings, so agents naturally slice
    (``res[-800:]``) and search (``'Goal reached: True' in res``) them;
    without these dunders both moves are ``TypeError``s that cost a
    recovery turn (and recur after compaction erases the lesson).
    """

    @property
    def text(self) -> str:
        """The full report as a plain string."""
        return repr(self)

    def __getitem__(self, key: Any) -> str:
        return repr(self)[key]

    def __contains__(self, item: str) -> bool:
        return item in repr(self)

    def __len__(self) -> int:
        return len(repr(self))


@dataclasses.dataclass(repr=False)
class ProbeResult(_StrLikeResult):
    """Outcome of one ``BeliefProbe.run`` call.

    Attributes mirror the ``evaluate_option_plan`` report: ``steps`` is
    a list of per-step dicts (``option``, ``num_actions``, ``failure``,
    ``added``, ``deleted``, ``subgoals_missing`` - the step's ``->
    {atoms}`` annotations that did NOT hold in the post-state (the
    forward-pass divergence signal), ``image`` - the saved post-step
    scene image path, if rendering is available; with ``contacts=True``
    also ``contacts`` - the step's contact-pair span lines), plus
    ``goal_reached`` and ``final_atoms``. ``notes`` carries caveats
    (ignored region annotations, horizon overruns). ``print(result)``
    renders the same step-by-step summary the tool prints.
    """
    steps: List[Dict[str, Any]]
    goal_reached: bool
    final_atoms: List[str]
    final_state: State
    notes: List[str] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        lines = []
        for i, s in enumerate(self.steps):
            line = f"Step {i}: {s['option']} ({s['num_actions']} actions)"
            if s["failure"]:
                line += f"\n  FAILURE: {s['failure']}"
            if s["added"] or s["deleted"]:
                line += (f"\n  Added: {{{', '.join(s['added'])}}}"
                         f"\n  Deleted: {{{', '.join(s['deleted'])}}}")
            if s.get("subgoals_missing"):
                line += ("\n  SUBGOAL NOT REACHED: "
                         f"{{{', '.join(s['subgoals_missing'])}}}")
            if "contacts" in s:
                line += (
                    "\n  Contacts: " +
                    ("; ".join(s["contacts"]) if s["contacts"] else "none"))
            lines.append(line)
        lines.append(f"Goal reached: {self.goal_reached}")
        lines.append(f"Final atoms: {{{', '.join(self.final_atoms)}}}")
        images = [s["image"] for s in self.steps if s.get("image")]
        if images:
            lines.append("Saved images (view with Read):")
            lines.extend(f"  {p}" for p in images)
        lines.extend(f"NOTE: {n}" for n in self.notes)
        return "\n".join(lines)


@dataclasses.dataclass(repr=False)
class ProbeTrialsResult(_StrLikeResult):
    """Outcome of one ``BeliefProbe.run(..., trials=N)`` call.

    ``trials`` holds one dict per trial (``goal_reached``,
    ``num_actions``, ``failure`` - ``None`` or ``"step {i} ({option}):
    {reason}"``; ``planner_seed`` - the motion-planner seed the trial
    ran at, reproducible via ``run(plan, seed=..., fresh=True)`` - a
    plain ``run(plan, seed=...)`` executes on the WARM shared env, a
    different substrate that reads optimistic; with
    ``solved=True`` also ``solved``/``reward`` from the task evaluator,
    ``None`` when the verdict errored). ``successes`` counts
    goal-reaching trials. The current state is NOT advanced - repeated
    trials are a measurement, not a navigation step.
    """
    trials: List[Dict[str, Any]]
    successes: int
    fresh_env_per_trial: bool
    notes: List[str] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        n = len(self.trials)
        env_note = ("fresh physics env + varied motion-planner seed per "
                    "trial - the rate estimates real execution reliability"
                    if self.fresh_env_per_trial else
                    "shared session env - trials are correlated, treat the "
                    "rate as optimistic")
        scored = [t for t in self.trials if t.get("solved") is not None]
        headline = f"Trials: {self.successes}/{n} reached the goal"
        if scored:
            n_solved = sum(1 for t in scored if t["solved"])
            headline += (f", {n_solved}/{n} scored solved=True by the task "
                         f"evaluator")
        lines = [f"{headline} ({env_note})"]
        for i, t in enumerate(self.trials):
            seed_tag = (f" (planner seed {t['planner_seed']})"
                        if t.get("planner_seed") is not None else "")
            if t["failure"]:
                line = f"  trial {i + 1}{seed_tag}: FAILED - {t['failure']}"
            elif t["goal_reached"]:
                line = (f"  trial {i + 1}{seed_tag}: goal reached "
                        f"({t['num_actions']} actions)")
            else:
                line = (f"  trial {i + 1}{seed_tag}: goal NOT reached "
                        f"({t['num_actions']} actions)")
            if t.get("solved") is not None:
                line += (f" - evaluator: solved={t['solved']}, "
                         f"reward={t['reward']:.2f}")
            lines.append(line)
        if scored and any(t["goal_reached"] and not t["solved"]
                          for t in scored):
            lines.append(
                "  goal atoms held but the evaluator scored a non-solve: "
                "this route's success did not certify under the task's "
                "rules. Either the rules reject the route (re-read the "
                "goal's rules and plan a different route) or the success "
                "is too marginal to reproduce reliably - same-route "
                "parameter tuning rarely flips this verdict.")
        lines.extend(f"NOTE: {n_}" for n_ in self.notes)
        return "\n".join(lines)


@dataclasses.dataclass(repr=False)
class ProbeSweepResult(_StrLikeResult):
    """Outcome of one ``BeliefProbe.run(..., physics_sweep=True)`` call.

    ``points`` holds one dict per sweep point in ascending-sigma order
    (``params`` - the physical-parameter override dict, ``None`` for
    the fitted-values reference point run first; ``goal_reached``,
    ``num_actions``, ``failure``). ``successes`` counts goal-reaching
    points. Each point runs once on a fresh env at the BASE
    motion-planner seed, so its outcome is a deterministic measurement
    of the plan at those physical parameters - a failing point is a
    hole in the design's success band, not noise. The current state is
    NOT advanced.
    """
    points: List[Dict[str, Any]]
    successes: int
    notes: List[str] = dataclasses.field(default_factory=list)

    def __repr__(self) -> str:
        n = len(self.points)
        lines = [
            f"Physics sweep: {self.successes}/{n} points reached the goal "
            "(fresh env + base planner seed per point - each point is a "
            "deterministic measurement, not a sample)"
        ]
        for p in self.points:
            desc = ("fitted params" if p["params"] is None else ", ".join(
                f"{k}={v:.4g}" for k, v in sorted(p["params"].items())))
            if p["failure"]:
                line = f"  {desc}: FAILED - {p['failure']}"
            elif p["goal_reached"]:
                line = f"  {desc}: goal reached ({p['num_actions']} actions)"
            else:
                line = (f"  {desc}: goal NOT reached "
                        f"({p['num_actions']} actions)")
            lines.append(line)
        if self.successes < n:
            lines.append(
                "  the plan fails INSIDE the identified-parameter "
                "uncertainty range. The real environment may sit at any "
                "of these points (success can be non-monotonic: passing "
                "neighbors do NOT cover the points between them), and the "
                "capture gate re-runs this same sweep - add design margin "
                "until every point passes before submitting.")
        lines.extend(f"NOTE: {n_}" for n_ in self.notes)
        return "\n".join(lines)


@dataclasses.dataclass(repr=False)
class ProbeSuggestResult(_StrLikeResult):
    """Outcome of ``BeliefProbe.suggest_probes``: per annotated step, the
    agent's own parameters' ensemble disagreement and the top alternatives."""
    suggestions: List[Any]
    notes: List[str]

    def __repr__(self) -> str:
        lines = [
            "Probe suggestions - ensemble disagreement on each step's "
            "subgoal atoms (0 = every posterior member agrees on the "
            "outcome, 1 = the members split; higher = a measurement that "
            "teaches the model more). These are SUGGESTIONS: your "
            "submitted parameters execute exactly as written, so adopt an "
            "alternative only by writing it into your sketch, and only on "
            "a step whose failure the episode can afford."
        ]
        for s in self.suggestions:
            objs = ", ".join(s.objects)
            head = f"  step {s.step_idx} {s.option_name}({objs}) -> " \
                f"{{{', '.join(s.subgoal_atoms)}}}:"
            if s.nominal_params is None:
                head += " no proposal;"
            elif not s.nominal_feasible:
                head += (f" proposed {_fmt_params(s.nominal_params)} does "
                         "NOT establish the subgoal;")
            else:
                per = ", ".join(f"{a}={v:.2f}"
                                for a, v in s.nominal_per_atom.items())
                head += (f" proposed {_fmt_params(s.nominal_params)} "
                         f"disagreement {s.nominal_score:.2f} ({per});")
            head += f" {s.n_feasible}/{s.n_draws} draws feasible"
            lines.append(head)
            if not s.candidates:
                lines.append("      no feasible alternative found")
            for rank, (params, score, per_atom) in enumerate(s.candidates, 1):
                per = ", ".join(f"{a}={v:.2f}" for a, v in per_atom.items())
                lines.append(f"      alt {rank}: {_fmt_params(params)} "
                             f"disagreement {score:.2f} ({per})")
        if not self.suggestions:
            lines.append("  (no annotated step with continuous parameters)")
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        return "\n".join(lines)


@dataclasses.dataclass(repr=False)
class ProbeRefineResult(_StrLikeResult):
    """Outcome of one ``BeliefProbe.refine`` call.

    ``verdict`` states exactly what a SUCCESS certifies - ``executed``
    (every step established its subgoal annotation; the task goal was
    never checked), ``goal-reached`` (the goal atoms also held at the
    last step), or ``evaluator-solved`` (the task evaluator additionally
    scored the rollout as a solve) - so "SUCCESS" alone is never read
    as more than it means. ``plan_lines`` holds one line per sketch
    step with the refined params filled in (``[?]`` for steps the
    search never refined) - paste them into ``sim.run`` or
    ``evaluate_option_plan``. ``near_miss`` is the deepest validation
    failure (step index, the exact params that got furthest, and why
    they failed), also populated on timeout/exhaustion. ``note``
    carries caveats.
    """
    success: bool
    reason: str
    total_samples: int
    step_samples: List[int]
    plan_lines: List[str]
    near_miss: Optional[Dict[str, Any]]
    note: str = ""
    verdict: str = ""

    def __repr__(self) -> str:
        lines = [
            f"Refinement {'SUCCESS' if self.success else 'FAILURE'} "
            f"({self.reason}): {self.total_samples} samples, per-step "
            f"{self.step_samples}"
        ]
        if self.verdict:
            lines.append(f"Verdict: {self.verdict}")
        lines.append("Plan (refined params; [?] = never refined):")
        lines.extend(f"  {l}" for l in self.plan_lines)
        if self.near_miss is not None:
            lines.append(f"Deepest near-miss: step "
                         f"{self.near_miss['step_idx']} "
                         f"{self.near_miss['option']} - "
                         f"{self.near_miss['reason']}")
        if self.note:
            lines.append(f"NOTE: {self.note}")
        return "\n".join(lines)


@dataclasses.dataclass(repr=False)
class ProbeGatherResult(_StrLikeResult):
    """Outcome of one ``BeliefProbe.gather`` call.

    One status line per handle; the full reports stay on the handles
    (``print(handle.result)`` for a done handle, ``handle.error`` for a
    failed one). ``pending`` handles are still running - keep them and
    gather again later. ``stale`` flags done handles whose rollout ran
    under an older ``simulator.py`` than the current file.
    """
    done: List[Any]
    pending: List[Any]
    stale: List[Any]
    current_tag: Optional[str]

    def __repr__(self) -> str:
        lines = [
            f"Gather: {len(self.done)} done, {len(self.pending)} pending."
        ]
        for h in self.done:
            head = f"  handle {h.index}"
            if not h.ok:
                first = str(h.error).splitlines()[0] if h.error else "unknown"
                lines.append(f"{head}: FAILED - {first}")
                continue
            r = h.result
            if isinstance(r, ProbeTrialsResult):
                summary = (f"trials {r.successes}/{len(r.trials)} reached "
                           "the goal")
            elif isinstance(r, ProbeResult):
                summary = f"goal_reached={r.goal_reached}"
            else:
                summary = type(r).__name__
            lines.append(f"{head}: {summary} - print(handle.result) for the "
                         "full report")
        for h in self.pending:
            lines.append(f"  handle {h.index}: still running")
        if self.stale:
            idxs = ", ".join(str(h.index) for h in self.stale)
            lines.append(
                f"NOTE: handle(s) {idxs} ran under an OLDER simulator.py "
                "than the current file - their results describe the model "
                "as of their launch, not your latest edits.")
        return "\n".join(lines)


class BeliefProbe:
    """Stateful exploration handle over the belief simulator.

    Typical loop::

        print(sim.task())      # goal + objects + init details (no state
                               # change; pass task_idx in synthesis)
        sim.reset()                                   # true task init
        sim.reset(mods={"domino_1": {"x": 0.46}})     # modified copy
        sid = sim.snapshot()
        out = sim.run("Push(robot:robot, domino_0:domino)[0.05, 0.05]\\n"
                      "Wait(robot:robot)[]")
        print(out)             # per-step outcomes
        sim.state("domino_1")  # full-precision features
        sim.render("after_push")
        sim.restore(sid)
        # Search params for a suffix from here (nothing is captured):
        print(sim.refine(
            "Place(robot:robot)[0.46, 1.32, 0.55, -1.0] ~ [0.05, 0.05, "
            "0.0, 0.5] -> {SomeSubgoal(domino_1:domino)}"))

    The "current state" is just a ``State`` object; ``run`` executes from
    it (the option model resets the sim env from that state, exactly as
    ``evaluate_option_plan`` does from a task init) and advances it to
    the rollout's final state.
    """

    # Distinct deterministic rng streams per instance (see refine).
    _next_instance_id = 0

    def __init__(self, ctx: "ToolContext"):
        self._ctx = ctx
        self._state: Optional[State] = None
        self._base_task: Optional[Task] = None
        # True when the base task is the solve-time "current" task (a
        # plain reset()/first use) rather than an explicit train
        # task_idx. The instance outlives individual solve queries, so
        # current-task probes must follow ctx.current_task when the
        # harness re-points it (see _require_state).
        self._tracking_current_task = False
        self._snapshots: Dict[int, Tuple[State, bool]] = {}
        self._next_snapshot_id = 1
        self._refine_calls = 0
        self._instance_id = BeliefProbe._next_instance_id
        BeliefProbe._next_instance_id += 1
        # True while the current state IS the task's unmodified initial
        # state (no mods, no rollout since reset). Gates require_solved:
        # the task evaluator's staging rules reference the true init, so
        # a verdict from any other start would be silently wrong.
        self._pristine = False
        # Lazy fork registry for run_async (created on first use).
        self._async_registry: Optional[Any] = None

    # ── State control ────────────────────────────────────────────

    def _option_model(self) -> Any:
        """The option model probes execute against.

        Solve sessions bind the deployed belief model via
        ``ctx.option_model``. Synthesis sessions install
        ``ctx.probe_option_model_provider`` instead - a lazy builder
        over the candidate ``simulator.py`` the agent is editing (fresh
        MCMC fit, cached until the file changes) - so probes always
        exercise the latest belief model, never the stale pre-synthesis
        one (real physics on cycle 1: a live-env leak).
        """
        provider = self._ctx.probe_option_model_provider
        if provider is not None:
            return provider()
        return self._ctx.option_model

    def reset(self,
              task_idx: Optional[int] = None,
              mods: Optional[Modifications] = None) -> "BeliefProbe":
        """Set the current state to a task's initial state, optionally with
        object-feature overrides applied to a copy.

        ``task_idx`` indexes the train tasks; ``None`` uses the current
        solve-time task. Returns ``self`` so calls chain.
        """
        ctx = self._ctx
        _check_time_budget(ctx)
        if (task_idx is None and ctx.probe_option_model_provider is not None):
            # Synthesis session: "current task" is a solve-time pointer
            # and may dangle at whatever task the harness touched last -
            # silently probing it is the stale-current-task bug class.
            raise ValueError(
                "During synthesis there is no current solve task; pass "
                "task_idx explicitly, e.g. sim.reset(task_idx=0).")
        if task_idx is not None:
            if not 0 <= task_idx < len(ctx.train_tasks):
                raise ValueError(f"Invalid task_idx {task_idx}. Available: "
                                 f"0-{len(ctx.train_tasks) - 1}")
            task = ctx.train_tasks[task_idx]
        elif ctx.current_task is not None:
            task = ctx.current_task
        else:
            raise ValueError("No task_idx given and no current task set.")
        state = task.init
        if mods:
            mod_list = self._normalize_mods(mods)
            state, _, err = apply_state_modifications(state, mod_list)
            if err:
                raise ValueError(err)
        else:
            state = state.copy()
        self._state = state
        self._base_task = task
        self._tracking_current_task = task_idx is None
        self._pristine = not mods
        return self

    @staticmethod
    def _normalize_mods(mods: Modifications) -> List[Dict[str, Any]]:
        if isinstance(mods, dict):
            return [{
                "object": name,
                "features": feats
            } for name, feats in mods.items()]
        return list(mods)

    def snapshot(self) -> int:
        """Bank a copy of the current state; returns an id for restore.

        Snapshots rewind the ABSTRACT state only: replays from a
        restored snapshot still share the session env's solver state and
        velocity residuals, so repeated restore-and-run trials are
        correlated with each other (and read optimistic vs. a fresh
        env). For honest reliability estimates pass ``trials=N`` to
        ``run``, which uses a fresh physics env per trial when the
        session provides one.
        """
        sid = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._snapshots[sid] = (self._require_state().copy(), self._pristine)
        return sid

    def restore(self, snapshot_id: int) -> "BeliefProbe":
        """Set the current state back to a snapshot."""
        self._require_state()  # follow a task change before restoring
        if snapshot_id not in self._snapshots:
            raise ValueError(f"Unknown snapshot id {snapshot_id}. "
                             f"Available: {sorted(self._snapshots)}")
        state, pristine = self._snapshots[snapshot_id]
        self._state = state.copy()
        self._pristine = pristine
        return self

    def drop(self, snapshot_id: int) -> "BeliefProbe":
        """Free a banked snapshot (they otherwise live for the session)."""
        self._snapshots.pop(snapshot_id, None)
        return self

    def clear_snapshots(self) -> "BeliefProbe":
        """Free every banked snapshot."""
        self._snapshots.clear()
        return self

    # ── Introspection ────────────────────────────────────────────

    def task(self, task_idx: Optional[int] = None) -> str:
        """Describe a task: goal (NL preferred), initial atoms, objects, and
        initial-state details.

        ``task_idx`` indexes the train tasks; ``None`` (solve sessions
        only) describes the current solve-time task. Purely
        informational - does not change the probe's current state. Use
        ``reset(task_idx)`` + ``render()`` for the scene image.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.tools.digests import render_task_digest
        ctx = self._ctx
        if task_idx is None and ctx.probe_option_model_provider is not None:
            raise ValueError(
                "During synthesis there is no current solve task; pass "
                "task_idx explicitly, e.g. sim.task(0).")
        if task_idx is not None:
            if not 0 <= task_idx < len(ctx.train_tasks):
                raise ValueError(f"Invalid task_idx {task_idx}. Available: "
                                 f"0-{len(ctx.train_tasks) - 1}")
            task = ctx.train_tasks[task_idx]
            label: Union[int, str] = task_idx
        elif ctx.current_task is not None:
            task = ctx.current_task
            label = "(current solve task)"
        else:
            raise ValueError("No task_idx given and no current task set.")
        # The is_goal_state/train_tasks query hint only makes sense in
        # synthesis sessions, whose exec namespace binds those names.
        return render_task_digest(
            task,
            label,
            ctx.predicates,
            include_goal_query_hint=(ctx.probe_option_model_provider
                                     is not None))

    def fit(self,
            traj_idxs: Optional[List[int]] = None,
            fixed: Optional[Dict[str, float]] = None,
            path: Optional[str] = None) -> str:
        """Fit the candidate simulator's parameters and return the report.

        Synthesis sessions only (at solve time the deployed belief
        model is fixed). With no arguments this is the CANONICAL fit -
        the same fit the probe deploys for ``run``/``refine``, on the
        full data - and, when ``simulator.py`` declares
        ``PHYSICAL_PARAMS``, the identified physical values are applied
        to the planning base env. Passing ``traj_idxs`` (fit only those
        trajectories' data) or ``fixed`` (pin parameters at given
        values) makes the fit EXPLORATORY: a diagnostic report only -
        nothing is published or applied, and the probe keeps running
        the canonical fit. On the system-ID path ``traj_idxs`` is a
        cross-trajectory consistency check (subset fits that disagree
        mean heterogeneous data); ``fixed`` is rejected there - pin by
        narrowing the param's bounds in PHYSICAL_PARAMS. Reports SSE at
        init vs post-fit, fitted values with deltas, and (system-ID
        path) per-parameter identifiability. MCMC - the expensive probe
        call; use deliberately.
        """
        ctx = self._ctx
        _check_time_budget(ctx)
        provider = ctx.probe_fit_provider
        if provider is None:
            raise RuntimeError(
                "sim.fit is unavailable in this session: fitting happens "
                "during learning; the solve-time belief model is fixed.")
        return provider(path=path, traj_idxs=traj_idxs, fixed=fixed)

    def residuals(self,
                  max_transitions: int = 100,
                  abs_tol: float = 1e-4,
                  rel_tol: float = 1e-3,
                  num_worst_examples: int = 3,
                  fit_params: bool = False,
                  path: Optional[str] = None,
                  rollout: bool = False,
                  sweep_num_points: int = 6,
                  sweep_params: Optional[Union[str, List[str]]] = None,
                  phys_params: Optional[Dict[str, float]] = None) -> str:
        """Per-feature residual report for the current simulator rules.

        Synthesis sessions only. Loads RESIDUAL_RULES fresh from
        ``simulator.py`` and reports, per feature in RESIDUAL_FEATURES,
        mismatch counts, mean/max abs error, improvement over the
        no-rule baseline (negative = the rules hurt), and the worst-N
        example transitions - the fast inner loop for finding where the
        rules disagree with the data. Uses init_value params unless
        ``fit_params=True`` (MCMC-fits first; diagnostic only, nothing
        published). Tolerance: ``|pred - obs| > rel_tol * |obs| +
        abs_tol``. Snapshots the simulator file and tags the report
        ``[cycle_XXX_vers_YYY]``. ``path`` scores a DIFFERENT simulator
        file than the canonical one - use it to compare rule-set
        candidates side by side.

        ``rollout=True`` switches to the OPEN-LOOP report: free-running
        replay of each recorded trajectory (errors compound, which the
        teacher-forced default structurally cannot see). It is the only
        residual view that can implicate or exonerate a physical
        parameter; consult it before deciding the ``PHYSICAL_PARAMS``
        declaration either way. Two mutually exclusive opt-ins probe
        the parameters themselves:

        * ``sweep_params`` - a list of env-registry parameter names, or
          ``"all"`` - sweeps each named parameter alone across its
          plausible range (``sweep_num_points`` candidates each) and
          reports whether the data is explained materially better at
          another value. Slow: one fresh-env rollout per candidate per
          motion segment, so name the suspects (or pass ``"all"`` once
          before the declaration decision) rather than sweeping in a
          loop.
        * ``phys_params`` - ``{name: value}`` - scores the data at ONE
          hypothesized point and reports the SSE ratio vs the baseline;
          the cheap composable primitive for your own targeted sweeps.
        """
        ctx = self._ctx
        _check_time_budget(ctx)
        provider = ctx.probe_residuals_provider
        if provider is None:
            raise RuntimeError(
                "sim.residuals is unavailable in this session: residual "
                "reports are a learning diagnostic; there is no candidate "
                "simulator to score at solve time.")
        return provider(max_transitions=max_transitions,
                        abs_tol=abs_tol,
                        rel_tol=rel_tol,
                        num_worst_examples=num_worst_examples,
                        fit_params=fit_params,
                        path=path,
                        rollout=rollout,
                        sweep_num_points=sweep_num_points,
                        sweep_params=sweep_params,
                        phys_params=phys_params)

    def state(
        self,
        obj_name: Optional[str] = None
    ) -> Union[Dict[str, Dict[str, float]], Dict[str, float]]:
        """Full-precision feature dict of the current state.

        ``state()`` -> ``{obj: {feat: value}}`` for all objects;
        ``state("domino_1")`` -> that object's ``{feat: value}``.
        """
        cur = self._require_state()

        def _features(obj: Any) -> Dict[str, float]:
            return {
                feat: float(cur.get(obj, feat))
                for feat in obj.type.feature_names
            }

        if obj_name is not None:
            # Sweep loops call the single-object form per iteration;
            # keep it O(one object), not O(scene).
            for obj in cur:
                if obj.name == obj_name:
                    return _features(obj)
            raise ValueError(f"Unknown object '{obj_name}'. Available: "
                             f"{sorted(o.name for o in cur)}")
        return {obj.name: _features(obj) for obj in sorted(cur, key=str)}

    def atoms(self) -> List[str]:
        """Sorted ground atoms true in the current state."""
        cur = self._require_state()
        preds = self._ctx.predicates
        return [str(a) for a in sorted(utils.abstract(cur, preds))]

    def render(
            self,
            label: str = "probe",
            annotations: Optional[List[Dict[str,
                                            Any]]] = None) -> Optional[str]:
        """Render the current state; returns the saved image path.

        ``annotations`` overlays temporary geometry for this render
        only: dicts with ``type`` of ``marker``
        (``position`` [x, y, z]), ``line`` (``from``/``to``), or
        ``rectangle`` (``min_corner``/``max_corner``), plus optional
        ``color`` [r, g, b], ``size``, and ``label`` text. Use it to
        mark candidate positions, offsets, and reference points on the
        staged scene.
        """
        # pylint: disable=import-outside-toplevel
        import pybullet as pb

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        cur = self._require_state()
        ctx.test_call_id += 1
        if annotations:
            if ctx.env is None:
                raise ValueError("No environment available for rendering.")
            ctx.env._set_state(cur)  # pylint: disable=protected-access
            physics_id = ctx.env._physics_client_id  # pylint: disable=protected-access
            debug_ids: List[int] = []
            try:
                for ann in annotations:
                    debug_ids.extend(draw_pybullet_annotation(ann, physics_id))
                img = render_pybullet_image(ctx, f"probe_{label}")
            finally:
                # Remove only the drawn bodies (never the env's own),
                # also on a bad-annotation error mid-draw.
                for body_id in debug_ids:
                    try:
                        pb.removeBody(body_id, physicsClientId=physics_id)
                    except Exception:  # pylint: disable=broad-except
                        pass
        else:
            img = render_pybullet_image(ctx, f"probe_{label}", state=cur)
        saved = img.get("saved_path") if img else None
        # The saved filename is prefixed (iter/task/test counters), so a
        # guessed path never matches - print the real one even when the
        # agent doesn't print the return value (audited runs lost turns
        # Read-ing guessed names).
        if saved:
            print(f"Saved scene image (view with Read): {saved}")
        return saved

    # ── Execution ────────────────────────────────────────────────

    def _parse_sketch(self, plan_text: str) -> Any:
        """Parse ``plan_text`` against the current state.

        Returns ``(probe_task, sketch_steps, all_predicates, notices)``;
        the probe task starts at the current state and carries no
        evaluator, and ``notices`` lists parse caveats to surface (e.g.
        region annotations ignored because ground samplers are off).
        Same grammar and parser as ``evaluate_option_plan`` (``~ [w]``
        search regions included).
        """
        # pylint: disable=import-outside-toplevel
        from predicators.agent_sdk import bilevel_sketch

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        cur = self._require_state()
        assert self._base_task is not None
        probe_task = dataclasses.replace(self._base_task,
                                         init=cur,
                                         evaluator=None)

        all_options = ctx.options
        all_predicates = ctx.predicates
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in cur)

        gs_fns, gs_err = load_ground_sampler_fns(ctx)
        if gs_err is not None:
            raise ValueError(gs_err)
        notices: List[str] = []
        sketch_steps = bilevel_sketch.parse_sketch_from_text(
            plan_text,
            probe_task,
            predicates=all_predicates,
            options=all_options,
            types=types,
            parse_continuous_params=True,
            strict=True,
            parse_ground_samplers=RefinementConfig.from_cfg().ground_samplers,
            ground_sampler_fns=gs_fns or None,
            notices=notices)
        if not sketch_steps:
            raise ValueError(
                "Parsed empty plan. Each line must be "
                "`Option(obj:type, ...)[params]` with a known option, typed "
                "object refs, and exact params in `[]`.")
        status = ctx.probe_param_status
        if status and status.startswith("UNFITTED"):
            notices.append(f"PARAMS {status}.")
        return probe_task, sketch_steps, all_predicates, notices

    def _require_solved_evaluator(self, flag: str) -> Any:
        """Precondition gate shared by ``run(solved=True)`` and
        ``refine(require_solved=True)``, so the two evaluator-scored surfaces
        can never drift apart.

        Returns the task evaluator. Raises when the current state is not
        the task's unmodified initial state (the evaluator's rules
        reference the true init, so any other start would give silently-
        wrong verdicts) or when the task defines no evaluator.
        """
        if not self._pristine:
            raise ValueError(
                f"{flag} needs the task's unmodified initial state (call "
                "reset() with no mods first, before any state-advancing "
                "run()): the evaluator's rules reference the true init, so "
                "a verdict from a modified or advanced state would be "
                "silently wrong.")
        assert self._base_task is not None
        evaluator = self._base_task.evaluator
        if evaluator is None:
            raise ValueError(f"{flag}: this task defines no task evaluator.")
        return evaluator

    def run(
        self,
        plan_text: str,
        render: bool = True,
        trials: int = 1,
        solved: bool = False,
        contacts: bool = False,
        physics_sweep: bool = False,
        seed: Optional[int] = None,
        fresh: bool = False,
    ) -> Union[ProbeResult, ProbeTrialsResult, ProbeSweepResult]:
        """Execute an option plan from the current state.

        ``plan_text`` uses the same grammar as ``evaluate_option_plan``:
        one option per line, ``Option(obj:type, ...)[params]`` with
        exact continuous params (``[]`` for none); ``-> {atoms}``
        subgoal annotations are optional but CHECKED - each step's
        report lists any annotated atoms that did not hold in its
        post-state, which makes a single continuous ``run`` of a
        refined plan the forward-validation pass (``refine`` resets
        state between options and resamples, so a refine-pass that
        diverges here means a rule is more permissive than the env).
        Advances the current state
        to the rollout's final state (``restore`` a snapshot to rewind).
        Like ``evaluate_option_plan``, each step's post-state is
        rendered to a saved image whose path lands in the step report;
        pass ``render=False`` inside tight sweep loops to skip that.
        Exploratory only: results are never captured.

        ``trials=N`` (N > 1) runs the SAME plan N times and returns a
        ``ProbeTrialsResult`` with the per-trial outcomes and success
        count - use it to estimate a plan's reliability instead of
        hand-rolled repeat loops. Each trial runs on a freshly
        constructed physics env when the session provides one (repeats
        on the shared env - including snapshot/restore replays - share
        solver state and velocity residuals, so they are correlated
        and read optimistic). With ``trials > 1`` nothing is rendered
        and the current state is NOT advanced.

        ``solved=True`` (needs ``trials`` >= 2 and the task's UNMODIFIED
        initial state - plain ``reset()``, no rollout since) additionally
        scores every trial with the TASK EVALUATOR, reporting per-trial
        ``solved``/``reward``. Reaching the goal atoms is NOT the same as
        being scored a solve - the evaluator can reject a goal-reaching
        route - so check ``solved`` counts here BEFORE submitting via
        ``evaluate_option_plan`` instead of discovering rejections one
        submission at a time.

        ``contacts=True`` (single-run mode only, ``trials=1``) records
        every physical contact during the rollout and reports, per step,
        which robot links touched which objects and which object pairs
        came into contact, with low-level action spans. Use it to check
        WHAT actually caused motion - e.g. whether a topple was driven
        by the object you pushed or by the robot's body brushing the
        scene.

        ``physics_sweep=True`` (its own mode - incompatible with
        ``trials``/``solved``/``contacts``) re-runs the plan once at
        each point of a grid spanning the +-1-sigma uncertainty range
        of the identified physical parameters (the SAME points the
        capture gate checks), each on a fresh env at the base
        motion-planner seed, plus once at the fitted values, and
        returns a ``ProbeSweepResult`` with per-point outcomes. The
        fitted values carry real uncertainty and the true environment
        may sit anywhere in the range - and success can be
        NON-MONOTONIC in a physical parameter (a cascade that topples
        just above and just below some friction can fail exactly at
        it), so ``trials=`` at the fitted values CANNOT see this. A
        design near a feasibility boundary (e.g. the minimal block
        count) is exactly where such holes live: sweep it and add
        margin until EVERY point passes before submitting, instead of
        discovering PARAM-SENSITIVE rejections one capture at a time.
        Rollouts are deterministic per point, so each point costs one
        rollout and its outcome is a measurement, not a sample. The
        current state is NOT advanced and nothing is rendered.

        ``seed=S`` overrides the base motion-planner seed for this call.
        Trials report the planner seed each ran at (trial ``i`` runs at
        ``S + i``; without ``seed=`` at ``base + i``), and
        ``evaluate_option_plan``'s validation rollouts report theirs the
        same way. A single run (``trials=1``) executes entirely at
        ``S``; a physics sweep runs every point at ``S`` instead of the
        base.

        SUBSTRATE: a default single run executes on the WARM shared
        session env from the probe's current state - a feature for
        mid-exploration probing (after ``reset(mods=...)`` or a partial
        rollout), but a different physics substrate from trials,
        validation rollouts, and the real episode, which all run on a
        freshly constructed env. A warm pass at a seed where a fresh
        trial failed is evidence of shared-env optimism, not seed luck.
        ``fresh=True`` (single-run mode only) reproduces a failed
        trial's substrate exactly: ``run(plan, seed=<reported seed>,
        fresh=True)`` runs once on a fresh env from the parsed start
        state, without advancing the shared session state (per-step
        renders are skipped - the renderer draws the shared env).
        """
        # pylint: disable-next=import-outside-toplevel
        import contextlib

        # pylint: disable-next=import-outside-toplevel
        import numpy as np

        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk import bilevel_sketch
        # pylint: disable-next=import-outside-toplevel
        from predicators.settings import CFG
        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}")
        if physics_sweep and (trials > 1 or solved or contacts):
            raise ValueError(
                "physics_sweep=True is its own mode: it varies the PHYSICS "
                "per rollout (one deterministic rollout per parameter "
                "point), while trials/solved/contacts measure the plan at "
                "the fitted values. Run them as separate calls.")
        if solved and trials < 2:
            raise ValueError(
                "solved=True needs trials >= 2: a single shared-env rollout "
                "is a correlated sample, so a one-off verdict would read "
                "optimistic. Use trials>=2 (fresh env per trial), or "
                "refine(..., require_solved=True) to search for "
                "evaluator-solved params.")
        if contacts and trials > 1:
            raise ValueError(
                "contacts=True is only available in single-run mode "
                "(trials=1): per-contact recording is a diagnostic for one "
                "rollout, not a statistic.")
        if fresh and (trials > 1 or physics_sweep):
            raise ValueError(
                "fresh=True applies to single runs only: trials and "
                "physics_sweep already run every rollout on a freshly "
                "constructed env.")
        if fresh and contacts:
            raise ValueError(
                "fresh=True cannot record contacts: contact recording "
                "binds to the shared session env. Run contacts=True "
                "without fresh (the warm substrate), or diagnose the "
                "fresh-substrate failure via the step report alone.")
        ctx = self._ctx
        _check_time_budget(ctx)
        ctx.test_call_id += 1
        probe_task, sketch_steps, all_predicates, notices = \
            self._parse_sketch(plan_text)
        grounded: List[Any] = []
        for st in sketch_steps:
            params = (st.initial_params if st.initial_params is not None else
                      np.array([], dtype=np.float32))
            grounded.append(
                st.option.ground(list(st.objects),
                                 np.asarray(params, dtype=np.float32)))

        report_preds = ctx.predicates

        def _horizon_note(total_actions: int) -> Optional[str]:
            budget = utils.real_episode_step_budget(ctx.phase)
            if total_actions > budget:
                return (f"the rollout used {total_actions} low-level steps, "
                        f"more than the real episode's step budget "
                        f"({budget}) - the real executor would run out "
                        "of steps, so shorten or speed up the plan.")
            return None

        evaluator: Any = None
        if solved:
            evaluator = self._require_solved_evaluator("solved=True")

        if physics_sweep:
            fresh_scope = (ctx.validation_env_scope
                           if ValidationConfig.from_cfg().fresh_env
                           and ctx.validation_env_scope is not None
                           and ctx.probe_option_model_provider is None else
                           None)
            if fresh_scope is None:
                raise ValueError(
                    "physics_sweep=True needs the session's fresh-env "
                    "scope (unavailable here): perturbing the shared "
                    "session env would leak the perturbation into every "
                    "later call.")
            provider = ctx.physics_margin_provider
            sweep_points = list(provider() or []) if provider else []
            if not sweep_points:
                raise ValueError(
                    "physics_sweep=True, but no identified physical "
                    "parameters with nonzero posterior width are deployed "
                    "this cycle, so there is no uncertainty range to "
                    "sweep. Use trials= to measure execution reliability "
                    "at the current physics instead.")
            point_dicts: List[Dict[str, Any]] = []
            all_points: List[Optional[Dict[str, float]]] = \
                [None] + sweep_points
            # Rebound so the nested function sees the narrowed
            # (non-Optional) type - the None-check above cannot
            # propagate into a closure.
            sweep_scope = fresh_scope

            def _one_point(
                    point: Optional[Dict[str, float]]) -> Dict[str, Any]:
                # Base planner seed at every point (no
                # decorrelated_rollout_seed), matching the capture
                # gate's margin rollouts: with the seed held fixed, an
                # outcome flip between points is attributable to the
                # physics perturbation alone.
                with (sweep_scope() if point is None else sweep_scope(
                        physical_overrides=point)), \
                        absolute_rollout_seed(seed):
                    model = self._option_model()
                    r = bilevel_sketch.execute_plan_forward(
                        probe_task,
                        grounded,
                        model,
                        predicates=all_predicates,
                        sketch=sketch_steps,
                        stop_on_failure=True)
                point_failure: Optional[str] = None
                if r.first_failure_idx is not None:
                    fs = r.steps[r.first_failure_idx]
                    point_failure = (f"step {r.first_failure_idx} "
                                     f"({_fmt_option(fs.option)}): "
                                     f"{fs.failure_reason or 'not initiable'}")
                return {
                    "params": point,
                    "goal_reached": r.goal_reached,
                    "num_actions": sum(s.num_actions for s in r.steps),
                    "failure": point_failure,
                }

            point_prefetched = prefetch_parallel(
                [functools.partial(_one_point, pt) for pt in all_points],
                "probe physics sweep")
            try:
                for point_idx, point in enumerate(all_points):
                    _check_time_budget(ctx)
                    _count_rollout(ctx)
                    pre = point_prefetched[point_idx]
                    point_dicts.append(
                        pre if pre is not None else _one_point(point))
            except ProbeBudgetExceeded as e:
                # Same salvage rule as trials mode: completed points are
                # sim time living in the return value, not stdout.
                if not point_dicts:
                    raise
                notices.append(
                    f"time budget expired after {len(point_dicts)}/"
                    f"{len(all_points)} sweep points - the remaining "
                    f"points were skipped ({e})")
            budget = utils.real_episode_step_budget(ctx.phase)
            sweep_over = [
                p for p in point_dicts
                if p["goal_reached"] and p["num_actions"] > budget
            ]
            if sweep_over:
                notices.append(
                    f"{len(sweep_over)} goal-reaching sweep point(s) "
                    f"exceeded the real episode's step budget ({budget} "
                    "low-level steps) - the real executor would run out "
                    "of steps.")
            sweep_successes = sum(1 for p in point_dicts if p["goal_reached"])
            return ProbeSweepResult(point_dicts, sweep_successes, notices)

        if trials > 1:
            # Fresh physics per trial when the session provides the scope
            # (solve sessions do; a synthesis probe's candidate model has
            # its own env, which the scope does not manage). Same CFG gate
            # as evaluate_option_plan's validation rollouts, so the two
            # surfaces sample the same distribution.
            fresh_scope = (ctx.validation_env_scope
                           if ValidationConfig.from_cfg().fresh_env
                           and ctx.validation_env_scope is not None
                           and ctx.probe_option_model_provider is None else
                           None)
            trial_dicts: List[Dict[str, Any]] = []
            base_planner_seed = seed if seed is not None else CFG.seed

            def _one_trial(trial_idx: int) -> Dict[str, Any]:
                trial_solved: Optional[bool] = None
                trial_reward: Optional[float] = None
                coarse = False
                # decorrelated_rollout_seed: a fresh env alone gives
                # bit-identical repeats (motion planning reads the
                # constant CFG.seed), so without it N trials are one
                # effective sample. Entered inside the fresh scope so
                # env construction keeps the base seed.
                trial_inexact: List[str] = []
                with (fresh_scope() if fresh_scope is not None else
                      contextlib.nullcontext()), \
                        absolute_rollout_seed(seed), \
                        decorrelated_rollout_seed(trial_idx):
                    model = self._option_model()
                    collector = (_EvalStateCollector(model, probe_task.init)
                                 if evaluator is not None else None)

                    def _trial_on_step(i: int, outcome: Any) -> None:
                        # The first option's execution resets the
                        # (fresh) env to the parsed start state; any
                        # features that reset could not round-trip
                        # are a start-state reconstruction error, a
                        # failure source distinct from plan margin.
                        if i == 0:
                            env = getattr(model, "sim_env", None)
                            lossy = getattr(
                                env, "_last_unreconstructible_features",
                                None) or []
                            trial_inexact.extend(f"{obj.name}.{feat}"
                                                 for obj, feat in lossy)
                        if collector is not None:
                            collector.on_step(i, outcome)

                    r = bilevel_sketch.execute_plan_forward(
                        probe_task,
                        grounded,
                        model,
                        predicates=all_predicates,
                        sketch=sketch_steps,
                        on_step=_trial_on_step,
                        stop_on_failure=True)
                    # Score INSIDE the scope: the evaluator's
                    # certificate probes at the (fresh) env the
                    # rollout ran on, same as the capture path.
                    if (collector is not None and len(collector.states) > 1):
                        try:
                            verdict = evaluate_states_with(
                                evaluator,
                                collector.states,
                                collector.labels,
                                sim_env=getattr(model, "sim_env", None))
                            # Reward first: it is the only fallible
                            # conversion, so a malformed verdict is
                            # caught below with BOTH fields still None
                            # instead of crashing the loop and losing
                            # the completed trials.
                            trial_reward = float(verdict["reward"])
                            trial_solved = bool(verdict["solved"])
                            # Only a produced verdict can be coarse; an
                            # errored one must not trip the coarse
                            # caveat.
                            coarse = collector.coarse
                        except Exception as e:  # pylint: disable=broad-except
                            logging.debug("Trial evaluator verdict failed: %s",
                                          e)
                failure: Optional[str] = None
                if r.first_failure_idx is not None:
                    fs = r.steps[r.first_failure_idx]
                    failure = (f"step {r.first_failure_idx} "
                               f"({_fmt_option(fs.option)}): "
                               f"{fs.failure_reason or 'not initiable'}")
                total = sum(s.num_actions for s in r.steps)
                return {
                    "goal_reached": r.goal_reached,
                    "num_actions": total,
                    "failure": failure,
                    "solved": trial_solved,
                    "reward": trial_reward,
                    "verdict_coarse": coarse,
                    "planner_seed": base_planner_seed + trial_idx,
                    "inexact_start_features": sorted(set(trial_inexact)),
                }

            trial_prefetched = prefetch_parallel(
                [functools.partial(_one_trial, k) for k in range(trials)],
                "probe trials")
            try:
                for trial_idx in range(trials):
                    _check_time_budget(ctx)
                    _count_rollout(ctx)
                    pre = trial_prefetched[trial_idx]
                    trial_dicts.append(
                        pre if pre is not None else _one_trial(trial_idx))
            except ProbeBudgetExceeded as e:
                # Completed trials are minutes of sim time and live in the
                # RETURN VALUE, not stdout - discarding them on a mid-loop
                # budget stop would force re-simulating them. Return the
                # partial result instead; only an empty result re-raises.
                if not trial_dicts:
                    raise
                notices.append(
                    f"time budget expired after {len(trial_dicts)}/{trials} "
                    f"trials - the remaining trials were skipped ({e})")
            successes = sum(1 for t in trial_dicts if t["goal_reached"])
            if evaluator is not None:
                if any(t["verdict_coarse"] for t in trial_dicts):
                    notices.append(
                        "some verdicts are coarse (per-step states were "
                        "unavailable, scored on option-boundary states "
                        "only) - a coarse verdict can falsely reject.")
                if any(t["goal_reached"] and t["solved"] is None
                       for t in trial_dicts):
                    notices.append(
                        "the evaluator errored on some goal-reaching "
                        "trials, so their solved verdicts are missing.")
            budget = utils.real_episode_step_budget(ctx.phase)
            over = [
                t for t in trial_dicts
                if t["goal_reached"] and t["num_actions"] > budget
            ]
            if over:
                notices.append(
                    f"{len(over)} goal-reaching trial(s) exceeded the real "
                    f"episode's step budget ({budget} low-level steps) - "
                    "the real executor would run out of steps.")
            inexact = [t for t in trial_dicts if t["inexact_start_features"]]
            if inexact:
                feats = sorted(
                    {f
                     for t in inexact for f in t["inexact_start_features"]})
                shown = ", ".join(
                    feats[:8]) + ("..." if len(feats) > 8 else "")
                notices.append(
                    f"{len(inexact)}/{len(trial_dicts)} trials started from "
                    f"an inexactly reconstructed state (features: {shown}) "
                    "- failures may partly reflect start-state "
                    "reconstruction error, not plan margin.")
            return ProbeTrialsResult(trial_dicts, successes, fresh_scope
                                     is not None, notices)

        run_fresh_scope = None
        if fresh:
            run_fresh_scope = (ctx.validation_env_scope
                               if ValidationConfig.from_cfg().fresh_env
                               and ctx.validation_env_scope is not None
                               and ctx.probe_option_model_provider is None else
                               None)
            if run_fresh_scope is None:
                raise ValueError(
                    "fresh=True needs the session's fresh-env scope "
                    "(unavailable here - synthesis sessions run the "
                    "candidate model on its own env). Rerun without "
                    "fresh=True.")
            if render:
                render = False
                notices.append(
                    "fresh=True: per-step renders skipped (the renderer "
                    "draws the shared session env, which this rollout does "
                    "not touch).")
            notices.append(
                "fresh=True: rollout ran on a freshly constructed env from "
                "the parsed start state; the shared session state was NOT "
                "advanced.")

        step_dicts: List[Dict[str, Any]] = []

        def _on_step(i: int, outcome: Any) -> None:
            sig = _fmt_option(outcome.option)
            added: List[str] = []
            deleted: List[str] = []
            failure = outcome.failure_reason
            if not outcome.initiable:
                failure = failure or "not initiable"
            if outcome.post_state is not None:
                before = utils.abstract(outcome.pre_state, report_preds)
                after = utils.abstract(outcome.post_state, report_preds)
                added = [str(a) for a in sorted(after - before)]
                deleted = [str(a) for a in sorted(before - after)]
            if i == 0 and fresh:
                env = getattr(model, "sim_env", None)
                lossy = getattr(env, "_last_unreconstructible_features",
                                None) or []
                if lossy:
                    feats = ", ".join(
                        sorted({f"{obj.name}.{feat}"
                                for obj, feat in lossy}))
                    notices.append(
                        "the fresh env could not reconstruct the start "
                        f"state exactly (features: {feats}) - a failure "
                        "here may partly reflect start-state "
                        "reconstruction error, not plan margin.")
            # Same per-step audit image evaluate_option_plan saves; the
            # env already sits at the post-step state here.
            img = render_scene_image(
                ctx,
                f"probe_step_{i}_{outcome.option.name}") if render else None
            step_dicts.append({
                "option":
                sig,
                "num_actions":
                outcome.num_actions,
                "failure":
                failure,
                "added":
                added,
                "deleted":
                deleted,
                "subgoals_missing":
                [str(a) for a in sorted(outcome.subgoal_missing or set())],
                "image":
                img.get("saved_path") if img else None,
            })

        _count_rollout(ctx)
        model = self._option_model()
        # Contact recording rides on the physics env the option model
        # steps (its ``sim_env``, the same binding certificate probes
        # use); a candidate-simulator model without one degrades to a
        # notice instead of an error.
        contact_env: Any = None
        if contacts:
            env = getattr(model, "sim_env", None)
            if env is None or not hasattr(env, "start_contact_recording"):
                notices.append(
                    "contact recording unavailable: this session's option "
                    "model exposes no physics env.")
            else:
                contact_env = env
                contact_env.start_contact_recording()
        contact_events: List[Dict[str, Any]] = []
        if seed is not None:
            notices.append(f"rollout ran at planner seed {seed} (base seed "
                           f"overridden for this call)")
        try:
            with (run_fresh_scope() if run_fresh_scope is not None else
                  contextlib.nullcontext()), absolute_rollout_seed(seed):
                result = bilevel_sketch.execute_plan_forward(
                    probe_task,
                    grounded,
                    model,
                    predicates=all_predicates,
                    sketch=sketch_steps,
                    on_step=_on_step,
                    stop_on_failure=True)
        finally:
            if contact_env is not None:
                contact_events = contact_env.stop_contact_recording()
        if contact_env is not None:
            counted = sum(s["num_actions"] for s in step_dicts)
            recorded = contact_env.contact_steps_recorded
            if recorded != counted:
                notices.append(
                    f"contact recording saw {recorded} env steps but the "
                    f"step report counts {counted} actions - a partially "
                    "executed failing step reports 0 actions, so its "
                    "contacts are attached to the last step and spans "
                    "there are relative to that step's start.")
            _attach_step_contacts(contact_events, step_dicts)
        if not fresh:
            # A fresh-env rollout's final state is not the warm session
            # state; advancing to it would silently mix substrates.
            self._state = result.final_state
            self._pristine = False
        final_atoms = [
            str(a)
            for a in sorted(utils.abstract(result.final_state, report_preds))
        ]
        hn = _horizon_note(sum(s["num_actions"] for s in step_dicts))
        if hn is not None:
            notices.append(hn)
        return ProbeResult(step_dicts, result.goal_reached, final_atoms,
                           result.final_state, notices)

    def run_policy(
        self,
        trials: int = 1,
        seed: Optional[int] = None,
        source: Optional[str] = None,
    ) -> Union[ProbeResult, ProbeTrialsResult]:
        """Run ./policy.py closed-loop FROM THE CURRENT STATE.

        Policy-mode counterpart of ``run``: loads the sandbox's
        ``policy.py`` fresh (or executes ``source`` directly) and drives
        its ``get_option(state, memory)`` through the belief model with
        the same failure-surfacing semantics as ``evaluate_policy`` and
        the real executor - option failures land in
        ``memory['last_failure']`` and the policy is asked again;
        get_option bugs end the episode.

        Starting from the CURRENT probe state is the point: perturb or
        advance the state first (``reset(mods=...)``, a partial ``run``)
        and check that the policy RECOVERS from off-nominal states, not
        just the initial one. ``trials=N`` repeats the rollout from the
        SAME current state on fresh envs (fresh policy memory per
        trial). Never captures - deliver via ``evaluate_policy``. Also
        available in learn sessions (probing a candidate simulator);
        there it runs against the candidate model.
        """
        # pylint: disable=import-outside-toplevel
        import contextlib

        from predicators.agent_sdk.policy_execution import \
            build_policy_option_fn, execute_policy_forward
        from predicators.agent_sdk.tools.testing import _policy_source_path
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        _check_time_budget(ctx)
        ctx.test_call_id += 1
        if trials < 1:
            raise ValueError(f"trials must be >= 1, got {trials}")
        if source is None:
            path = _policy_source_path(ctx)
            if path is None or not os.path.isfile(path):
                raise ValueError(
                    "No ./policy.py found (and no `source` given). Write "
                    "`def get_option(state, memory): ...` there first.")
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        cur = self._require_state()
        assert self._base_task is not None
        probe_task = dataclasses.replace(self._base_task,
                                         init=cur,
                                         evaluator=None)
        all_options = ctx.options
        all_predicates = ctx.predicates
        types = set(ctx.types)
        for opt in all_options:
            types.update(opt.types)
        for pred in all_predicates:
            types.update(pred.types)
        types.update(o.type for o in cur)

        def _fresh_fn() -> Any:
            fn, err = build_policy_option_fn(source,
                                             probe_task,
                                             predicates=all_predicates,
                                             options=all_options,
                                             types=types)
            if err is not None or fn is None:
                raise ValueError(err or "policy failed to load")
            return fn

        max_opts = CFG.agent_policy_max_options
        notices: List[str] = []

        def _one_rollout(fn: Any) -> Any:
            model = self._option_model()
            return execute_policy_forward(probe_task,
                                          fn,
                                          model,
                                          predicates=all_predicates,
                                          max_policy_options=max_opts)

        if trials > 1:
            fresh_scope = (ctx.validation_env_scope
                           if ValidationConfig.from_cfg().fresh_env
                           and ctx.validation_env_scope is not None
                           and ctx.probe_option_model_provider is None else
                           None)
            trial_dicts: List[Dict[str, Any]] = []
            base_planner_seed = seed if seed is not None else CFG.seed
            try:
                for trial_idx in range(trials):
                    _check_time_budget(ctx)
                    _count_rollout(ctx)
                    with (fresh_scope() if fresh_scope is not None else
                          contextlib.nullcontext()), \
                            absolute_rollout_seed(seed), \
                            decorrelated_rollout_seed(trial_idx):
                        r = _one_rollout(_fresh_fn())
                    failure: Optional[str] = None
                    if r.policy_error is not None:
                        failure = f"policy error: {r.policy_error}"
                    elif r.first_failure_idx is not None and \
                            not r.goal_reached:
                        fs = r.steps[r.first_failure_idx]
                        failure = (f"step {r.first_failure_idx} "
                                   f"({_fmt_option(fs.option)}): "
                                   f"{fs.failure_reason}")
                    trial_dicts.append({
                        "goal_reached": r.goal_reached,
                        "num_actions": r.total_actions,
                        "failure": failure,
                        "planner_seed": base_planner_seed + trial_idx,
                        "inexact_start_features": [],
                    })
            except ProbeBudgetExceeded as e:
                # Same salvage rule as ``run``: completed trials live in
                # the return value, so a mid-loop budget stop returns the
                # partial result instead of discarding them.
                if not trial_dicts:
                    raise
                notices.append(
                    f"time budget expired after {len(trial_dicts)}/{trials} "
                    f"trials - the remaining trials were skipped ({e})")
            successes = sum(1 for t in trial_dicts if t["goal_reached"])
            return ProbeTrialsResult(trial_dicts, successes, fresh_scope
                                     is not None, notices)

        _count_rollout(ctx)
        step_dicts: List[Dict[str, Any]] = []

        def _on_step(_i: int, outcome: Any) -> None:
            failure = outcome.failure_reason
            if failure is not None:
                failure = (f"{failure} [surfaced to the policy as "
                           "memory['last_failure']]")
            added: List[str] = []
            deleted: List[str] = []
            if outcome.post_state is not None:
                before = utils.abstract(outcome.pre_state, ctx.predicates)
                after = utils.abstract(outcome.post_state, ctx.predicates)
                added = [str(a) for a in sorted(after - before)]
                deleted = [str(a) for a in sorted(before - after)]
            step_dicts.append({
                "option": _fmt_option(outcome.option),
                "num_actions": outcome.num_actions,
                "failure": failure,
                "added": added,
                "deleted": deleted,
                "subgoals_missing": [],
                "image": None,
            })

        with absolute_rollout_seed(seed):
            result = execute_policy_forward(probe_task,
                                            _fresh_fn(),
                                            self._option_model(),
                                            predicates=all_predicates,
                                            max_policy_options=max_opts,
                                            on_step=_on_step)
        if result.policy_error is not None:
            notices.append(f"POLICY ERROR ended the episode: "
                           f"{result.policy_error}")
        # A closed-loop probe advances the session state like `run`.
        self._state = result.final_state
        self._pristine = False
        final_atoms = [
            str(a)
            for a in sorted(utils.abstract(result.final_state, ctx.predicates))
        ]
        return ProbeResult(step_dicts, result.goal_reached, final_atoms,
                           result.final_state, notices)

    # ── Async rollouts (launch now, gather later) ────────────────

    def _model_tag(self) -> Optional[str]:
        """Identity of the current belief model, stamped on handles.

        The sandbox ``simulator.py`` mtime: cheap, monotone under the
        agent's edits, and enough to flag results that predate the
        current file.
        """
        sandbox = getattr(self._ctx, "sandbox_dir", None)
        if not sandbox:
            return None
        path = os.path.join(sandbox, "simulator.py")
        try:
            return f"simulator.py@{os.path.getmtime(path):.3f}"
        except OSError:
            return None

    def run_async(self, plan_text: str, **kwargs: Any) -> Any:
        """Launch ``run(plan_text, ...)`` in a forked child; return a handle
        immediately.

        Launch several independent runs, keep working (edit the model,
        fit, think - even across ``run_python`` calls, handles survive
        in the namespace), then collect with ``gather``. Two semantic
        differences from ``run``: the session state does NOT advance
        (the child executes on a forked copy - a measurement, not
        navigation), and rendering is unavailable (``render`` is forced
        off). The child snapshots the session AND the belief model at
        launch, so results describe the model as of the launch;
        ``gather`` flags results whose ``simulator.py`` is older than
        the current file. When parallel workers are disabled the run
        executes inline (under a snapshot, so the no-advance semantics
        hold) and the returned handle is already done.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.async_rollouts import \
            AsyncRolloutRegistry, CompletedRollout, async_rollouts_available
        # pylint: disable-next=import-outside-toplevel
        from predicators.settings import CFG
        ctx = self._ctx
        _check_time_budget(ctx)
        kwargs["render"] = False
        _count_rollout(ctx, max(1, int(kwargs.get("trials", 1))))
        tag = self._model_tag()
        workers = int(getattr(CFG, "agent_validation_parallel_workers", 0))
        if workers <= 1 or not async_rollouts_available():
            sid = self.snapshot()
            try:
                return CompletedRollout(result=self.run(plan_text, **kwargs),
                                        tag=tag)
            except Exception as e:  # pylint: disable=broad-except
                return CompletedRollout(error=f"{type(e).__name__}: {e}",
                                        tag=tag)
            finally:
                self.restore(sid)
                self.drop(sid)
        if self._async_registry is None:
            self._async_registry = AsyncRolloutRegistry(max_workers=workers)
        job = functools.partial(self.run, plan_text, **kwargs)
        return self._async_registry.launch(job, tag=tag)

    def gather(self,
               handles: Sequence[Any],
               timeout: Optional[float] = None) -> ProbeGatherResult:
        """Wait for ``run_async`` handles and summarize their statuses.

        Blocks up to ``timeout`` seconds (default 1500, under the
        session's per-call watchdog) while pumping queued launches.
        Details stay on the handles: ``print(handle.result)`` for a
        done one, ``handle.error`` for a failed one. Pending handles
        are still running - keep them and gather again later.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.async_rollouts import AsyncRollout
        _check_time_budget(self._ctx)
        live = [h for h in handles if isinstance(h, AsyncRollout)]
        if live and self._async_registry is not None:
            self._async_registry.gather(
                live, timeout=1500.0 if timeout is None else timeout)
        done = [h for h in handles if h.done()]
        pending = [h for h in handles if not h.done()]
        current = self._model_tag()
        stale = [
            h for h in done
            if h.tag is not None and current is not None and h.tag != current
        ]
        return ProbeGatherResult(done=done,
                                 pending=pending,
                                 stale=stale,
                                 current_tag=current)

    def suggest_probes(self,
                       sketch_text: str,
                       max_draws: int = 20,
                       top_k: int = 3) -> "ProbeSuggestResult":
        """Rank alternative continuous parameters by what they would teach.

        Rolls ``sketch_text`` forward FROM THE CURRENT STATE on your own
        parameters and, at every step annotated with ``-> {subgoals}``
        that has continuous parameters, draws up to ``max_draws``
        alternatives, keeps the ones that still establish the subgoal,
        and ranks them by the learned model's ensemble disagreement on
        those atoms (the posterior members' split on whether the
        subgoal holds). Returns the top ``top_k`` per step next to your
        proposal's own score. Pure advice: nothing you submit is
        changed by this - write an alternative into your sketch to run
        it. Needs the info-seeking ensemble (solve/explore sessions
        with ``agent_explorer_info_seeking``); otherwise the result
        says so.
        """
        # pylint: disable=import-outside-toplevel
        import numpy as np

        from predicators.agent_sdk import bilevel_sketch
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        _check_time_budget(ctx)
        scorer = ctx.atom_disagreement_fn
        if scorer is None:
            return ProbeSuggestResult([], [
                "no ensemble disagreement scorer is available in this "
                "session (info-seeking ensemble not installed), so "
                "alternatives cannot be ranked."
            ])
        probe_task, sketch_steps, all_predicates, notices = \
            self._parse_sketch(sketch_text)
        self._refine_calls += 1
        rng = np.random.default_rng(CFG.seed + 100003 *
                                    (self._instance_id + 1) +
                                    self._refine_calls)

        def _on_rollout() -> None:
            _check_time_budget(ctx)
            _count_rollout(ctx)

        suggestions, notes = bilevel_sketch.suggest_probes(
            probe_task,
            sketch_steps,
            self._option_model(),
            predicates=all_predicates,
            info_scorer=scorer,
            rng=rng,
            max_draws=max(1, int(max_draws)),
            top_k=max(1, int(top_k)),
            parameterized_samplers=ctx.parameterized_samplers or None,
            on_rollout=_on_rollout)
        return ProbeSuggestResult(suggestions, list(notices) + notes)

    def refine(self,
               sketch_text: str,
               timeout: float = 60.0,
               max_samples_per_step: Optional[int] = None,
               require_goal: bool = False,
               require_solved: bool = False) -> "ProbeRefineResult":
        """Backtracking parameter search for a sketch FROM THE CURRENT STATE.

        Same grammar as ``evaluate_option_plan``, and composable:
        refine a plan *suffix* from a snapshot where the
        prefix already executed, so the search budget goes to the step
        that matters instead of re-descending through the whole plan.
        Annotate each step's ``-> {subgoals}`` - success means every
        step established its annotation (set ``require_goal=True`` to
        also demand the task goal at the last step; the result's
        ``Verdict`` line states which of these a SUCCESS certifies).
        Each step's ``[params]`` seed the search (tried first, then
        sampled around); ``~ [w]`` half-width regions confine the
        sampling ONLY when ground samplers are enabled in the session
        config - otherwise the annotation is accepted but ignored (a
        NOTE says so) and sampling stays uniform. Does not advance the
        current state. Returns best-found params (also on TIMEOUT - the
        refined prefix is reported as far as it got) plus per-step
        sample counts and the deepest near-miss. Exploratory only:
        nothing is captured.

        ``require_solved=True`` (implies ``require_goal``) additionally
        gates final-step acceptance on the TASK EVALUATOR's public
        ``solved`` verdict, so the search rejects parameters that reach
        the goal atoms via a route the evaluator scores as a non-solve
        and keeps searching (the near-miss records such rejections).
        Only valid when the current state is the task's UNMODIFIED
        initial state (plain ``reset()``, no rollout since): the
        evaluator's rules reference the true init, so any other start
        would give silently-wrong verdicts.
        """
        # pylint: disable=import-outside-toplevel
        import numpy as np

        from predicators.agent_sdk import bilevel_sketch
        from predicators.settings import CFG

        # pylint: enable=import-outside-toplevel
        ctx = self._ctx
        _check_time_budget(ctx)
        probe_task, sketch_steps, all_predicates, notices = \
            self._parse_sketch(sketch_text)
        solved_check: Optional[Callable[[List[State], List[Any], bool],
                                        Tuple[bool, str]]] = None
        gate_ran = [False]
        gate_called = [False]
        if require_solved:
            require_goal = True
            evaluator = self._require_solved_evaluator("require_solved")
            # Same gate (and therefore same accept policy: coarse and
            # evaluator errors never block, non-terminated never blocks)
            # as evaluate_option_plan, so identical params can't get
            # contradictory verdicts across the two surfaces.
            inner_check = make_solved_check(
                evaluator, getattr(self._option_model(), "sim_env", None))

            def gated_solved_check(states: List[State], labels: List[Any],
                                   coarse: bool) -> Tuple[bool, str]:
                # Track whether the gate was consulted at all vs. with a
                # real (non-coarse) rollout: "never consulted" means no
                # candidate reached the goal atoms (the blocker is
                # upstream of scoring), while "only coarse" means the
                # option model exposed no per-step trajectories - the
                # two need different explanations (a shared message here
                # read as a system failure in run_20260717 audits).
                gate_called[0] = True
                if not coarse:
                    gate_ran[0] = True
                return inner_check(states, labels, coarse)

            solved_check = gated_solved_check

        if max_samples_per_step is None:
            max_samples_per_step = \
                RefinementConfig.from_cfg().max_samples_per_step
        self._refine_calls += 1
        # Deterministic but distinct from the solver's CFG.seed + attempt
        # streams and from other probe instances, so
        # "try a different random search" does not replay failed draws.
        rng = np.random.default_rng(CFG.seed + 100003 *
                                    (self._instance_id + 1) +
                                    self._refine_calls)
        outcome = bilevel_sketch.refine_sketch(
            probe_task,
            sketch_steps,
            self._option_model(),
            predicates=all_predicates,
            timeout=timeout,
            rng=rng,
            max_samples_per_step=max_samples_per_step,
            check_subgoals=True,
            check_final_goal=require_goal,
            run_id="probe",
            parameterized_samplers=ctx.parameterized_samplers or None,
            strip_latent_wait_targets=not ctx.latent_tracking_available,
            solved_check=solved_check)
        refined_plan, success = outcome.plan, outcome.success
        total_samples = outcome.total_samples
        step_samples = outcome.step_samples_cumulative
        reason = outcome.termination_reason or ("success"
                                                if success else "failure")
        plan_lines: List[str] = []
        for i, st in enumerate(sketch_steps):
            opt = refined_plan[i] if i < len(refined_plan) else None
            objs = ", ".join(f"{o.name}:{o.type.name}" for o in st.objects)
            params = _fmt_params(opt.params) if opt is not None else "[?]"
            atoms = sorted(str(a) for a in (st.subgoal_atoms or set()))
            atoms += sorted(f"NOT {a}"
                            for a in (st.subgoal_neg_atoms or set()))
            suffix = f" -> {{{', '.join(atoms)}}}" if atoms else ""
            plan_lines.append(f"{st.option.name}({objs}){params}{suffix}")
        near_miss: Optional[Dict[str, Any]] = None
        if outcome.deepest_failure is not None:
            df = outcome.deepest_failure
            near_miss = {
                "step_idx": df.step_idx,
                "option": _fmt_option(df.option),
                "reason": df.fail_reason,
            }
        note_parts = list(notices)
        if require_solved and not gate_ran[0]:
            if not gate_called[0]:
                note_parts.append(
                    "require_solved: no candidate reached the goal atoms at "
                    "the final step, so the task evaluator was never "
                    "consulted - the blocker is upstream of scoring (see "
                    "the near-miss).")
            else:
                note_parts.append(
                    "require_solved: the option model exposed no per-step "
                    "trajectories, so the evaluator gate only saw coarse "
                    "rollouts and this result is NOT certified.")
        # What a SUCCESS actually certifies, so it is never read as more
        # than it means (a bare "SUCCESS" was disproven with 5 fresh
        # rollouts in run_20260717_154753 seed2 - it only meant
        # "executed").
        if not success:
            verdict = ""
        elif require_solved and gate_ran[0]:
            verdict = ("evaluator-solved - the task evaluator scored a "
                       "full rollout of these params as a solve.")
        elif require_goal:
            verdict = ("goal-reached - the task's goal atoms held at the "
                       "final step; no evaluator verdict (use "
                       "require_solved=True from a pristine reset() for "
                       "that).")
        else:
            verdict = ("executed - every step established its subgoal "
                       "annotation; the task goal was NOT checked (set "
                       "require_goal=True to demand it).")
        return ProbeRefineResult(success, reason, total_samples, step_samples,
                                 plan_lines, near_miss, " ".join(note_parts),
                                 verdict)

    # ── Internals ────────────────────────────────────────────────

    def _require_state(self) -> State:
        """Current state, following the harness's task pointer.

        The probe instance outlives individual solve queries while the
        harness re-points ``ctx.current_task`` per task; silently
        probing the PREVIOUS task's layout (identical object names, so
        plans would parse and run) is the stale-current-task bug class.
        When the tracked current task changes, reset to the new task's
        init and drop the old task's snapshots (restoring one would mix
        tasks).
        """
        if self._state is None:
            self.reset()
        elif (self._tracking_current_task
              and self._ctx.current_task is not None
              and self._ctx.current_task is not self._base_task):
            self._snapshots.clear()
            self.reset()
        assert self._state is not None
        return self._state


def build_probe_namespace(ctx: "ToolContext") -> Dict[str, Any]:
    """The persistent ``run_python`` namespace (solve sessions).

    The probe facade, numpy, and the collected REAL trajectories as
    read-only evidence (``trajectories`` plus a ``describe_trajectory``
    digest helper) - so the agent can check the belief model against
    what actually happened. Deliberately NOT here: anything
    evaluator-shaped (``evaluate_trajectory`` scores arbitrary state
    sequences, which in a solve session is a free oracle for searching
    over sequences the evaluator likes without paying for physics;
    evaluator access stays behind the gated ``run(solved=True)`` /
    ``refine(require_solved=True)`` paths) and authoring bindings
    (``Predicate``, ``ParamSpec`` - synthesis-role, see
    ``_build_synthesis_exec_ns``).
    """
    # pylint: disable-next=import-outside-toplevel
    import numpy as np

    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.tools.digests import render_trajectory_digest
    all_trajs = list(ctx.offline_trajectories) + list(ctx.online_trajectories)

    def describe_trajectory(traj_idx: int,
                            include_states: bool = True,
                            include_atoms: bool = False,
                            max_timesteps: int = 10) -> str:
        return render_trajectory_digest(all_trajs,
                                        ctx.train_tasks,
                                        ctx.predicates,
                                        traj_idx,
                                        include_states=include_states,
                                        include_atoms=include_atoms,
                                        max_timesteps=max_timesteps)

    return {
        "sim": BeliefProbe(ctx),
        "BeliefProbe": lambda: BeliefProbe(ctx),
        "np": np,
        "trajectories": all_trajs,
        "describe_trajectory": describe_trajectory,
    }
