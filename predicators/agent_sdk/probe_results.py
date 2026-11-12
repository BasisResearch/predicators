"""The results of the ``sim`` probe's rollouts and plan searches.

An agent holds these objects: ``sim.run`` returns a :class:`ProbeResult`
or a :class:`ProbeTrialsResult`, ``sim.refine`` a
:class:`ProbeRefineResult`, ``sim.gather`` a :class:`ProbeGatherResult`
and ``sim.state()`` a :class:`_BareNameFallbackDict`. It can read their
class, their module's name and their documentation, in every arm, so this
module and its classes name what a result holds and nothing an arm may
leave out.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from predicators.structs import State


class _BareNameFallbackDict(Dict[str, Dict[str, float]]):
    """``state()``'s full dict: keyed ``name:type``, resolving BARE object
    names via ``__missing__``.

    Agent code persisted before the ``name:type`` keying (sandbox
    helpers, journal recipes replayed after a resume/requeue) indexes
    the full dict by bare name; resolving that here costs new code
    nothing and closes the same gap the single-object form's bare-name
    path already closes. Ambiguous bare names (two objects sharing a
    name across types - not constructible in current envs) fall through
    to the ordinary KeyError.
    """

    def __missing__(self, key: str) -> Dict[str, float]:
        matches = [k for k in self if k.split(":", 1)[0] == key]
        if len(matches) == 1:
            return self[matches[0]]
        raise KeyError(key)


class _StrLikeResult:
    """String conveniences shared by the probe result types.

    The results print like strings, so agents naturally slice
    (``res[-800:]``), search (``'Goal reached: True' in res``), and call
    string methods (``res.split('\\n')`` destroyed a 1200 s refine
    result in run_20260830_145216 - AttributeError, then a full
    identical re-run); without these both moves are errors that cost a
    recovery turn (and recur after compaction erases the lesson). Any
    ``str`` method not shadowed by a real attribute delegates to the
    rendered report, so the whole string API works.
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

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found normally (dataclass
        # fields and real methods take precedence). Delegate public str
        # methods (.split, .splitlines, .find, .lower, ...) to the
        # rendered report; anything else raises AttributeError as usual.
        if not name.startswith("_") and hasattr(str, name):
            return getattr(repr(self), name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")


@dataclasses.dataclass(repr=False)
class ProbeResult(_StrLikeResult):
    """Outcome of one ``sim.run`` call.

    ``steps`` is a list of per-step dicts (``option``, ``num_actions``,
    ``failure``, ``added``, ``deleted``, ``subgoals_missing`` - the
    step's ``-> {atoms}`` annotations that did NOT hold in the
    post-state (the forward-pass divergence signal), ``image`` - the
    saved post-step scene image path, if rendering is available; with
    ``contacts=True`` also ``contacts`` - the step's contact-pair span
    lines), plus ``goal_reached`` and ``final_atoms``. ``notes`` carries
    caveats (ignored region annotations, horizon overruns).
    ``print(result)`` renders the step-by-step summary.
    """
    steps: List[Dict[str, Any]]
    goal_reached: bool
    final_atoms: List[str]
    final_state: State
    notes: List[str] = dataclasses.field(default_factory=list)
    # The rollout's state sequence, start state first, per low-level
    # step where the option model records one (else per option): what
    # ``evaluate_trajectory`` scores, so a rollout can be scored or
    # inspected without rebuilding it by hand.
    states: List[State] = dataclasses.field(default_factory=list)
    # The state after each plan line, or None where the line did not run
    # cleanly (:func:`_line_states`).
    line_states: List[Optional[State]] = dataclasses.field(
        default_factory=list)
    # The option label of each transition in ``states``, and whether those
    # states are per option only: what a rehearsal scores this rollout on,
    # as it scores a draw (``_EvalStateCollector``).
    labels: List[Any] = dataclasses.field(default_factory=list)
    coarse: bool = False

    def __repr__(self) -> str:
        lines = []
        for i, s in enumerate(self.steps):
            line = f"Step {i}: {s['option']} ({s['num_actions']} actions)"
            if s["failure"]:
                line += f"\n  FAILURE: {s['failure']}"
                if s.get("partial_actions"):
                    line += (f"\n  It ran {s['partial_actions']} actions "
                             "before it stopped; the state after this run "
                             "is where it stopped, as after a real "
                             "failure.")
            if s.get("changed"):
                label = ("Changed before it stopped"
                         if s["failure"] else "Changed")
                line += f"\n  {label}: {', '.join(s['changed'])}"
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
    """Outcome of one ``sim.run(..., trials=N)`` call.

    ``trials`` holds one dict per trial (``goal_reached``,
    ``num_actions``, ``failure`` - ``None`` or ``"step {i} ({option}):
    {reason}"``; ``planner_seed`` - the motion-planner seed the trial
    ran at, reproducible via ``run(plan, seed=..., fresh=True)`` - a
    plain ``run(plan, seed=...)`` executes on the WARM shared env, a
    different substrate that reads optimistic; with
    ``solved=True`` also ``solved``/``reward`` from the task evaluator,
    ``None`` when the check is unavailable or errored), plus
    ``evaluation_status`` (accepted, rejected, unavailable, error, or
    not_requested) and a non-sensitive ``evaluation_error`` diagnostic.
    ``successes`` counts
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
                    "trial - reliability under this model and start state"
                    if self.fresh_env_per_trial else
                    "shared session env - trials are correlated, treat the "
                    "rate as optimistic; this session has no fresh-env "
                    "scope, so fresh=True is likewise unavailable")
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
            elif t.get("evaluation_status") in {"unavailable", "error"}:
                line += (f" - evaluator: {t['evaluation_status']} "
                         f"({t['evaluation_error']}); NOT certified")
            lines.append(line)
        replay_notes = [t["note"] for t in scored if t.get("note")]
        if replay_notes:
            lines.append(f"  verdict note: {replay_notes[0]}")
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
class ProbeRefineResult(_StrLikeResult):
    """Outcome of one ``sim.refine`` call.

    ``verdict`` states exactly what a SUCCESS certifies - ``executed``
    (every step established its subgoal annotation; the task goal was
    never checked), ``goal-reached`` (the goal atoms also held at the
    last step), or ``evaluator-solved`` (the task evaluator additionally
    scored the rollout as a solve) - so "SUCCESS" alone is never read
    as more than it means. ``plan_lines`` holds one line per sketch
    step with the refined params filled in (``[?]`` for steps the
    search never refined) - paste them into ``sim.run`` or
    ``skills_execute_plan``. ``near_miss`` is the deepest validation
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
    """Outcome of one ``sim.gather`` call.

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
