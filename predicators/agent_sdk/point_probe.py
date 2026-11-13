"""The ``sim`` of EMPIRIC from assets at one value per parameter.

Under ``continual_point_values_only`` the agent's ``sim`` is a
:class:`SimProbe`: the calls of the probe that this arm has, over the
agent's model at the fitted value of each parameter. A rollout runs once,
from the probe's current state. The class has no call built on a belief,
a draw or a stress test, so those names are absent from ``sim``, from
``dir(sim)`` and from ``help(sim)``, and an argument of one is an unknown
keyword. The agent can read this class's documentation, so it says what
each call does and names nothing else.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.probe_results import ProbeGatherResult, \
    ProbeRefineResult, ProbeResult, ProbeTrialsResult
from predicators.agent_sdk.rollout_probe import Modifications
from predicators.agent_sdk.tools.context import ToolContext
from predicators.structs import State

# The name the agent knows this arm's probe class by.
PROBE_NAME = "SimProbe"


class SimProbe:
    """Stateful exploration handle over your simulator.

    The model is your current ``simulator.py`` at the values of your last
    ``sim.fit()``. Typical loop::

        print(sim.task())      # the current level's goal, objects and
                               # init details (no state change)
        sim.reset()                                   # the level's start
        sim.reset(mods={"domino_1": {"x": 0.46}})     # modified copy
        sid = sim.snapshot()
        out = sim.run("Push(robot:robot, domino_0:domino)[0.05, 0.05]\\n"
                      "Wait(robot:robot)[]")
        print(out)             # per-step outcomes
        sim.state("domino_1")  # full-precision features
        sim.render("after_push")
        sim.restore(sid)

    The current state is a ``State`` object. ``run`` executes from it and
    advances it to the rollout's final state. Nothing here acts in the
    environment.
    """

    def __init__(self, ctx: ToolContext) -> None:
        self._probe = BeliefProbe(ctx)

    # -- State control -------------------------------------------------------

    def reset(self,
              task_idx: Optional[int] = None,
              mods: Optional[Modifications] = None,
              current: bool = False) -> "SimProbe":
        """Set the current state to a level's initial state, optionally with
        object-feature overrides applied to a copy.

        ``task_idx`` indexes ``train_tasks``, the levels reached so far
        (0 is the first, the last is the current level); ``None`` uses
        the current level. ``current=True`` starts instead from the last
        real observation the session recorded (every env tool result
        refreshes it), with inferred memory refreshed for the current
        model, so a rollout begins where the environment is now; the
        level's goal stays the task, and evaluator-scored modes
        (``solved``, ``require_solved``) are off, since they reference
        the task's true initial state. ``mods`` overrides features of
        the copy (``{'obj': {'x': 1.05}}``). Returns ``self`` so calls
        chain.
        """
        self._probe.reset(task_idx=task_idx, mods=mods, current=current)
        return self

    def check_restore(self) -> Dict[str, Any]:
        """Check current candidate memory/poses across fresh worlds, no ticks.

        This checks reconstruction, not whether the inferred state is
        true. Use after model edits and on held assemblies before
        trusting lifts. The robot's pose comes back through inverse
        kinematics, approximately by design, so its deltas are listed
        apart (``robot_deltas``) and do not reject the check.
        """
        return self._probe.check_restore()

    def snapshot(self) -> int:
        """Bank a copy of the current state; returns an id for restore.

        Snapshots rewind the ABSTRACT state only: replays from a
        restored snapshot still share the session env's solver state and
        velocity residuals, so repeated restore-and-run trials are
        correlated with each other (and read optimistic vs. a fresh
        env). For honest reliability estimates pass ``trials=N`` to
        ``run``, which uses a fresh physics env per trial.
        """
        return self._probe.snapshot()

    def restore(self, snapshot_id: int) -> "SimProbe":
        """Set the current state back to a snapshot."""
        self._probe.restore(snapshot_id)
        return self

    def drop(self, snapshot_id: int) -> "SimProbe":
        """Free a banked snapshot (they otherwise live for the session)."""
        self._probe.drop(snapshot_id)
        return self

    def clear_snapshots(self) -> "SimProbe":
        """Free every banked snapshot."""
        self._probe.clear_snapshots()
        return self

    # -- Introspection -------------------------------------------------------

    def task(self, task_idx: Optional[int] = None) -> str:
        """Describe a level: goal, initial atoms, objects, and initial-state
        details.

        ``task_idx`` indexes ``train_tasks``, the levels reached so far;
        ``None`` describes the current level. Purely informational: it
        does not change the current state. Use ``reset(task_idx)`` and
        ``render()`` for the scene image.
        """
        return self._probe.task(task_idx)

    def state(
        self,
        obj_name: Optional[str] = None
    ) -> Union[Dict[str, Dict[str, float]], Dict[str, float]]:
        """Full-precision feature dict of the current state.

        ``state()`` -> ``{"name:type": {feat: value}}`` for all objects;
        the keys use the same ``name:type`` form as atoms and plan
        lines. ``state("domino_1")`` and ``state("domino_1:domino")``
        both -> that object's ``{feat: value}``. The full dict also
        resolves bare-name lookups (``state()["domino_1"]``).
        """
        return self._probe.state(obj_name)

    def atoms(self) -> List[str]:
        """Sorted ground atoms true in the current state."""
        return self._probe.atoms()

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
        return self._probe.render(label, annotations)

    # -- The model against the recordings -------------------------------------

    def fit(self,
            traj_idxs: Optional[List[int]] = None,
            fixed: Optional[Dict[str, float]] = None,
            path: Optional[str] = None) -> str:
        """Fit the candidate simulator's parameters and return the report.

        With no arguments this is the CANONICAL fit, on all the
        recordings: the fitted values are applied to the model that
        ``run`` and ``refine`` roll out. Passing ``traj_idxs`` (fit only
        those trajectories' data) makes the fit EXPLORATORY: a
        diagnostic report only, with nothing published or applied.
        Subset fits that disagree mean the recordings differ in
        something the model does not capture. ``fixed`` is rejected for
        a model with declared physical parameters: pin a parameter by
        narrowing its declared range. The report gives the residual
        error at the declared and at the fitted values, each fitted
        value, and the parameters the recordings do not constrain. The
        fit is the expensive probe call; use it deliberately.
        """
        return self._probe.fit(traj_idxs=traj_idxs, fixed=fixed, path=path)

    def validate(self,
                 traj_idxs: Optional[List[int]] = None,
                 params: Optional[Dict[str, float]] = None) -> str:
        """Replay recorded actions at the current model's deployed parameters.

        It changes no parameter value, drops no recording and takes no
        real action. Returns errors for every selected full trajectory.
        ``traj_idxs`` indexes the available recordings, never unseen
        tasks. ``params`` optionally overrides named values for this
        diagnostic only, e.g. estimates from a subset of the recordings.
        Keep validation data out of parameter estimation and model
        design to call it held out. Errors are not a success certificate
        for a future plan.
        """
        return self._probe.validate(traj_idxs=traj_idxs, params=params)

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
        """Residual report for the current simulator.

        Loads the current simulator fresh from ``simulator.py`` and
        reports where it disagrees with the recorded data. Snapshots the
        simulator file and tags the report ``[cycle_XXX_vers_YYY]``.
        ``path`` scores a DIFFERENT simulator file than the canonical
        one: use it to compare candidates side by side.

        A model that is a ``RESIDUAL_ENV`` subclass gets the OPEN-LOOP
        report: free-running replay of each recorded trajectory, where
        errors compound. ``rollout=True`` asks for it by name. Two
        mutually exclusive options test the parameters themselves:

        * ``sweep_params`` - a list of parameter names, or ``"all"`` -
          replays the recordings with each named parameter alone at
          ``sweep_num_points`` values across its plausible range, and
          reports whether another value explains the data materially
          better. Slow: one fresh-env rollout per value per motion
          segment, so name the suspects.
        * ``phys_params`` - ``{name: value}`` - scores the data at ONE
          hypothesized point and reports the SSE ratio against the
          current values.

        For a rule-form model the default report is per transition: per
        feature in RESIDUAL_FEATURES, mismatch counts, mean/max abs
        error and the worst-N example transitions, with the tolerance
        ``|pred - obs| > rel_tol * |obs| + abs_tol``; ``fit_params=True``
        fits first, for this report only.
        """
        return self._probe.residuals(max_transitions=max_transitions,
                                     abs_tol=abs_tol,
                                     rel_tol=rel_tol,
                                     num_worst_examples=num_worst_examples,
                                     fit_params=fit_params,
                                     path=path,
                                     rollout=rollout,
                                     sweep_num_points=sweep_num_points,
                                     sweep_params=sweep_params,
                                     phys_params=phys_params)

    def predicates(self,
                   max_trajectories: int = 10,
                   max_groundings_per_predicate: int = 4) -> str:
        """Reload ``predicates.py`` and report milestone behaviour.

        Loads ``LEARNED_PREDICATES`` fresh from the file (snapshotting
        it into ``predicates_versions/``), validates each entry,
        installs the set so ``run`` / ``refine`` abstract with the
        draft, and reports, per predicate x grounding over the recorded
        trajectories: coverage (ever-true / ever-false), flip counts,
        and monotonicity (a milestone flips False->True once and stays
        true). Call it after every edit of ``predicates.py``.
        """
        return self._probe.predicates(
            max_trajectories=max_trajectories,
            max_groundings_per_predicate=max_groundings_per_predicate)

    # -- Execution -----------------------------------------------------------

    def run(self,
            plan_text: str,
            render: bool = True,
            trials: int = 1,
            solved: bool = False,
            contacts: bool = False,
            seed: Optional[int] = None,
            fresh: bool = False) -> Union[ProbeResult, ProbeTrialsResult]:
        """Execute an option plan from the current state.

        ``plan_text`` uses the same grammar as ``skills_execute_plan``:
        one option per line, ``Option(obj:type, ...)[params]`` with
        exact continuous params (``[]`` for none); ``-> {atoms}``
        subgoal annotations are optional but CHECKED: each step's report
        lists any annotated atoms that did not hold in its post-state,
        which makes a single continuous ``run`` of a refined plan the
        forward-validation pass (``refine`` resets state between options
        and resamples, so a refine-pass that diverges here means a rule
        is more permissive than the env). Advances the current state to
        the rollout's final state (``restore`` a snapshot to rewind).
        Each step's post-state is rendered to a saved image whose path
        lands in the step report; pass ``render=False`` inside tight
        loops to skip that. Exploratory only: nothing runs in the
        environment.

        ``trials=N`` (N > 1) runs the SAME plan N times and returns a
        ``ProbeTrialsResult`` with the per-trial outcomes and success
        count: use it to estimate a plan's reliability instead of
        hand-rolled repeat loops. Each trial runs on a freshly
        constructed physics env (repeats on the shared env, including
        snapshot/restore replays, share solver state and velocity
        residuals, so they are correlated and read optimistic). With
        ``trials > 1`` nothing is rendered and the current state is NOT
        advanced.

        ``solved=True`` (needs ``trials`` >= 2 and the task's UNMODIFIED
        initial state: plain ``reset()``, no rollout since) additionally
        scores every trial with the TASK EVALUATOR, reporting per-trial
        ``solved``/``reward``. Reaching the goal atoms is NOT the same
        as being scored a solve, since the evaluator can reject a
        goal-reaching route, so check ``solved`` counts here BEFORE
        executing the plan in the environment.

        ``contacts=True`` (single-run mode only, ``trials=1``) records
        every physical contact during the rollout and reports, per step,
        which robot links touched which objects and which object pairs
        came into contact, with low-level action spans. Use it to check
        WHAT actually caused motion, e.g. whether a topple was driven by
        the object you pushed or by the robot's body brushing the scene.

        ``seed=S`` overrides the base motion-planner seed for this call.
        Trials report the planner seed each ran at (trial ``i`` runs at
        ``S + i``; without ``seed=`` at ``base + i``). A single run
        (``trials=1``) executes entirely at ``S``.

        SUBSTRATE: a default single run executes on the WARM shared
        session env from the current state. That suits mid-exploration
        probing (after ``reset(mods=...)`` or a partial rollout), but it
        is a different physics substrate from trials and the real
        episode, which run on a freshly constructed env. A warm pass at
        a seed where a fresh trial failed is evidence of shared-env
        optimism, not seed luck. ``fresh=True`` (single-run mode only)
        reproduces a failed trial's substrate exactly: ``run(plan,
        seed=<reported seed>, fresh=True)`` runs once on a fresh env
        from the parsed start state, without advancing the shared
        session state (per-step renders are skipped, since the renderer
        shows the shared env).
        """
        if fresh and trials > 1:
            raise ValueError(
                "fresh=True applies to single runs only: trials already run "
                "every rollout on a freshly constructed env.")
        result = self._probe.run(plan_text,
                                 render=render,
                                 trials=trials,
                                 solved=solved,
                                 contacts=contacts,
                                 seed=seed,
                                 fresh=fresh)
        assert isinstance(result, (ProbeResult, ProbeTrialsResult))
        return result

    def run_policy(
        self,
        trials: int = 1,
        seed: Optional[int] = None,
        source: Optional[str] = None,
    ) -> Union[ProbeResult, ProbeTrialsResult]:
        """Run ./policy.py closed-loop FROM THE CURRENT STATE.

        Policy-mode counterpart of ``run``: loads the sandbox's
        ``policy.py`` fresh (or executes ``source`` directly) and drives
        its ``get_option(state, memory)`` through your simulator. Option
        failures land in ``memory['last_failure']`` and the policy is
        asked again; get_option bugs end the episode.

        Starting from the CURRENT state is the point: perturb or advance
        the state first (``reset(mods=...)``, a partial ``run``) and
        check that the policy RECOVERS from off-nominal states, not just
        the initial one. ``trials=N`` repeats the rollout from the SAME
        current state on fresh envs (fresh policy memory per trial).
        Nothing runs in the environment.
        """
        return self._probe.run_policy(trials=trials, seed=seed, source=source)

    def run_async(self, plan_text: str, **kwargs: Any) -> Any:
        """Launch ``run(plan_text, ...)`` in a forked child; return a handle
        immediately.

        Launch several independent runs, keep working (edit the model,
        think; handles survive in the namespace across ``run_python``
        calls), then collect with ``gather``. Two differences from
        ``run``: the session state does NOT advance (the child executes
        on a forked copy, a measurement and not navigation), and
        rendering is unavailable (``render`` is forced off). The child
        snapshots the session AND the model at launch, so results
        describe the model as of the launch; ``gather`` flags results
        whose ``simulator.py`` is older than the current file. Without
        parallel workers the run executes inline (under a snapshot, so
        the state still does not advance) and the returned handle is
        already done.
        """
        # The arguments are run's: an unknown one fails here as it does
        # there.
        inspect.signature(self.run).bind(plan_text, **kwargs)
        if kwargs.get("fresh") and int(kwargs.get("trials", 1)) > 1:
            raise ValueError(
                "fresh=True applies to single runs only: trials already run "
                "every rollout on a freshly constructed env.")
        return self._probe.run_async(plan_text, **kwargs)

    def gather(self,
               handles: Any,
               timeout: Optional[float] = None) -> ProbeGatherResult:
        """Wait for ``run_async`` handles and summarize them.

        Returns one status line per handle; read a done handle's
        ``.result`` and a failed one's ``.error``. ``timeout`` (seconds)
        bounds the wait: handles still running are listed as pending, so
        keep them and gather again later. Results that ran under an
        older ``simulator.py`` than the current file are flagged.
        """
        return self._probe.gather(handles, timeout=timeout)

    def refine(self,
               sketch_text: str,
               timeout: float = 60.0,
               max_samples_per_step: Optional[int] = None,
               require_goal: bool = False,
               require_solved: bool = False) -> ProbeRefineResult:
        """Backtracking parameter search for a sketch FROM THE CURRENT STATE.

        Same grammar as ``skills_execute_plan``, and composable: refine
        a plan *suffix* from a snapshot where the prefix already
        executed, so the search budget goes to the step that matters
        instead of re-descending through the whole plan. Annotate each
        step's ``-> {subgoals}``: success means every step established
        its annotation (set ``require_goal=True`` to also demand the
        task goal at the last step; the result's ``Verdict`` line states
        which of these a SUCCESS certifies). Each step's ``[params]``
        seed the search (tried first, then sampled around). Does not
        advance the current state. Returns best-found params (also on
        TIMEOUT: the refined prefix is reported as far as it got) plus
        per-step sample counts and the deepest near-miss. Exploratory
        only: nothing runs in the environment.

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
        return self._probe.refine(sketch_text,
                                  timeout=timeout,
                                  max_samples_per_step=max_samples_per_step,
                                  require_goal=require_goal,
                                  require_solved=require_solved)


def reward_model(score: Callable[..., Dict[str, Any]]) -> Callable[..., Any]:
    """The ``evaluate_trajectory`` of this arm's ``run_python`` over ``score``,
    the harness's reward model: the arguments this arm has, and documentation
    that names nothing else.

    It is defined here because the agent can read the name of the module
    a function comes from.
    """

    def evaluate_trajectory(states: Sequence[State],
                            actions: Optional[Sequence[Any]] = None,
                            task_idx: Optional[int] = None) -> Dict[str, Any]:
        """Score ``states`` with a level's reward model.

        ``task_idx`` indexes ``train_tasks``, the levels reached so far;
        omit it for the current level.
        ``states``: the sequence, ``states[t]`` before action ``t``.
        ``actions``: the recorded ``Action`` objects, or one label per
        transition, ``(option_name, (object_name, ...), (param, ...))``,
        ``None`` for a transition you do not attribute to a skill; omit
        for no labels. A rule that replays an action replays the labeled
        one with its parameters, and its canonical action when the
        sequence carries none. Returns ``{"reward", "solved", "note"}``;
        ``note`` says what a replaying rule simulated and on which
        substrate ("" when nothing was replayed). On a rollout of your
        simulator, or a sequence you assembled, the substrate is your
        simulator at its current fit.
        """
        return score(states, actions, task_idx)

    return evaluate_trajectory
