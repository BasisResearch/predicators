"""Synthesis-session tools for sim learning (create_synthesis_tools)."""
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators.agent_sdk.synthesis_backend import SynthesisBackend
from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths
from predicators.agent_sdk.tools.snapshots import _ArtifactSnapshotter


@dataclasses.dataclass(frozen=True)
class SynthesisToolkit:
    """What ``create_synthesis_tools`` builds for one synthesis session.

    ``tools`` are the MCP tools to attach; ``fit_runner`` and
    ``residuals_runner`` are the ``sim.fit`` / ``sim.residuals``
    backends (installed as ``ToolContext.probe_fit_provider`` /
    ``probe_residuals_provider``). All three share this session's
    snapshotter, so every report carries consistent
    ``[cycle_XXX_vers_YYY]`` tags.
    """
    tools: list
    fit_runner: Callable[..., str]
    residuals_runner: Callable[..., str]


def create_synthesis_tools(
    exec_ns: Dict[str, Any],
    base_pred_triples: list,
    inferred_process_features: Dict[str, List[str]],
    simulator_file: str,
    versions_dir: str,
    approach: Optional[SynthesisBackend] = None,
    sandbox_dir: Optional[str] = None,
    sandbox_dir_for_agent: Optional[str] = None,
    cycle_index_provider: Optional[Callable[[], int]] = None,
) -> SynthesisToolkit:
    """Create the sim-learning synthesis agent's tool surface.

    Returns a :class:`SynthesisToolkit` with ``tools = [run_python]``
    plus the ``fit_runner`` / ``residuals_runner`` behind ``sim.fit``
    and ``sim.residuals``. Plan validation is hand-composed by the
    agent in ``run_python`` (``sim.fit()`` then ``sim.refine`` then a
    continuous ``sim.run`` pass), so there is no separate refinement
    tool.

    The agent's source-of-truth for the simulator is the file at
    ``simulator_file`` (which it edits with ``Write`` / ``Edit``). The
    fit and residuals backends each ``exec`` that file fresh into an
    isolated namespace per call and read ``PROCESS_RULES``,
    ``PARAM_SPECS``, ``PROCESS_FEATURES`` from it — no namespace state
    leaks across iterations. Before loading, every call also snapshots
    the current contents into ``versions_dir`` as
    ``cycle_XXX_vers_YYY_simulator.py`` (``XXX`` from
    ``cycle_index_provider()``, ``YYY`` resetting per
    ``create_synthesis_tools`` call) so the full history of evaluated
    versions is preserved across cycles; identical-content calls reuse
    the prior snapshot. Each report is prefixed with the version tag
    (``[cycle_XXX_vers_YYY]``).

    * ``run_python`` — executes arbitrary Python in a persistent
      namespace pre-loaded with trajectory data (and, in synthesis
      sessions, the candidate probe ``sim``). It does **not** define
      rules — write ``simulator.py`` for that.
    * ``fit_runner`` (not a tool; bound as ``sim.fit``) — SSE of the
      current ``PROCESS_RULES`` at init_value params, plus post-fit
      SSE and fitted values; the joint rollout system-ID path when
      ``PHYSICAL_PARAMS`` is declared; exploratory ``traj_idxs`` /
      ``fixed`` variants that publish nothing.
    * ``residuals_runner`` (not a tool; bound as ``sim.residuals``) —
      per-feature breakdown of where the current rules disagree with
      observations: mismatch counts, mean/max abs error, comparison
      to the no-rule baseline, and worst-N example transitions per
      feature.

    Args:
        exec_ns: Persistent namespace for ``run_python``. Should
            contain ``trajectories``, ``np``, ``ParamSpec``.
        base_pred_triples: ``(s_base, action, s_next_obs)`` triples
            with the base step already advanced — eval/test consume
            ``s_base`` directly so no live env is needed.
        inferred_process_features: Data-driven default scope used
            when the agent hasn't declared ``PROCESS_FEATURES`` in
            ``simulator.py`` yet.
        simulator_file: Host path to the canonical simulator file
            the agent edits. Synthesis tools ``exec`` this file
            fresh on every call.
        versions_dir: Directory to write per-call snapshots into
            (created on first use).
        approach: ``AgentSimLearningApproach`` instance, used by the
            rollout system-ID fit path (raw trajectories, fit env,
            applying identified params). If ``None``, that path
            returns an error.
        sandbox_dir: Host path to the agent's sandbox root.  When set,
            ``run_python`` spills oversize output to
            ``<sandbox_dir>/tool_outputs/run_python/`` instead of
            letting the agent SDK truncate and dump it to
            ``~/.claude/projects/.../tool-results/``.  When ``None``,
            output is always returned inline.
        sandbox_dir_for_agent: Path prefix the agent sees for
            ``sandbox_dir`` (e.g. ``"."`` for local sandbox or
            ``"/sandbox"`` for docker).  Used only when building the
            human-readable path included in the spilled-output message.
        cycle_index_provider: Callable returning the current online
            learning cycle (1-indexed). Read at snapshot time so the
            same tools instance reflects later cycle bumps. If ``None``,
            cycle defaults to 0 (still valid; produces
            ``cycle_000_vers_YYY``).
    """
    # pylint: disable=import-outside-toplevel
    import traceback  # pylint: disable=redefined-outer-name,reimported
    from collections import defaultdict

    from claude_agent_sdk import tool as _sdk_tool
    tool = _make_coercing_tool(_sdk_tool)

    from predicators.code_sim_learning.fit_space import ParamSpec
    from predicators.code_sim_learning.fitting import compute_sse, \
        compute_sse_recurrent, fit_rule_parameters, \
        fit_rule_parameters_latent
    from predicators.code_sim_learning.identifiability import \
        format_identifiability, identifiability_report, \
        select_trustworthy_params
    from predicators.code_sim_learning.physical_sysid import \
        fit_params_rollout_trimmed
    from predicators.code_sim_learning.rollout_env import \
        physical_param_anchors
    from predicators.code_sim_learning.rollout_objective import \
        compute_rollout_sse
    from predicators.code_sim_learning.trajectory_prep import \
        compute_residual_scaling
    from predicators.code_sim_learning.utils import apply_rules, \
        has_latent_rules, iter_feature_residuals, read_latent_init, \
        read_physical_param_specs, read_simulator_components, \
        rollout_predictions, stamp_physical_spec_scales

    # pylint: enable=import-outside-toplevel

    _snapshotter = _ArtifactSnapshotter(
        live_file=simulator_file,
        versions_dir=versions_dir,
        artifact_name="simulator",
        cycle_index_provider=cycle_index_provider,
        missing_file_hint=("Use Write to create it with PROCESS_RULES, "
                           "PARAM_SPECS, PROCESS_FEATURES."),
    )
    # Spill oversize output from the synthesis tools into the sandbox,
    # so nothing is dumped to ``~/.claude/projects/.../tool-results/``.
    # (``run_python``'s own spill lives in ``_make_python_exec_tool``.)
    _text = _make_spilling_text_result(sandbox_dir,
                                       agent_prefix=sandbox_dir_for_agent)

    def _snapshot_and_load(
            path: str) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        """Snapshot ``path`` then exec it into a fresh namespace.

        Returns ``(rules, specs, features, latent_init, physical_specs,
        version_tag, error_msg)``; ``error_msg`` is ``None`` on success.
        ``latent_init`` is the optional ``LATENT_INIT`` export (``None``
        for fully- observable simulators) — the synthesis tools need it
        to score recurrent (5-arg) rules through the latent-threaded
        path. ``physical_specs`` is the optional ``PHYSICAL_PARAMS``
        export (system identification); when present, a physics-only
        artifact is valid and missing rules/specs default to empty
        lists. Snapshots are deduped by SHA256, so repeated calls on
        unchanged content reuse the prior ``cycle_XXX_vers_YYY`` tag.
        """
        raw, version_tag, err = _snapshotter.snapshot(path)
        if err is not None:
            return None, None, None, None, None, None, err
        assert raw is not None and version_tag is not None
        ns: Dict[str, Any] = {"np": np, "ParamSpec": ParamSpec}
        try:
            exec(raw.decode("utf-8"), ns)  # pylint: disable=exec-used
        except Exception:  # pylint: disable=broad-except
            return None, None, None, None, None, version_tag, (
                f"[{version_tag}] Error executing {path}:\n"
                f"{_scrub_host_paths(traceback.format_exc())}")
        rules, specs, features = read_simulator_components(ns)
        latent_init = read_latent_init(ns)
        physical_specs = read_physical_param_specs(ns)
        if rules is None:
            if not physical_specs:
                return None, None, None, None, None, version_tag, (
                    f"[{version_tag}] PROCESS_RULES missing or empty in "
                    f"{path}.")
            rules = []
        if specs is None:
            if not physical_specs:
                return None, None, None, None, None, version_tag, (
                    f"[{version_tag}] PARAM_SPECS missing or empty in "
                    f"{path}.")
            specs = []
        return (rules, specs, features, latent_init, physical_specs,
                version_tag, None)

    def _groups_for(triples: list) -> List[List[Tuple[Any, Any, Any]]]:
        """Slice flat base-pred triples into per-trajectory groups.

        Recurrent rules thread their latent block within a trajectory,
        so scoring/residuals must regroup the flat triples the same way
        the engine does. Reuses the bound approach's grouping (keyed off
        the same ``_fit_trajectories`` cache the engine uses); falls
        back to a single group when no approach is bound or the lengths
        don't line up — correct for the common single-demo case.
        """
        if approach is not None and hasattr(approach,
                                            "_group_triples_by_trajectory"):
            grouped = approach._group_triples_by_trajectory(  # pylint: disable=protected-access
                triples)
            if grouped:
                return grouped
        return [triples]

    def _evaluate_rollout_fit(rules: list, rule_specs: list,
                              physical_specs: list, latent_init: Any,
                              process_features: Dict[str, List[str]],
                              scope_note: str, version_tag: str) -> str:
        """Joint physical+rule system-ID fit on free-running rollouts.

        Reached from ``run_fit`` (``sim.fit``) when the artifact
        declares ``PHYSICAL_PARAMS``. Needs the bound approach for the
        raw (states, actions) trajectories and the dedicated headless
        fit env. On success the identified physical values are applied
        in place to the approach's planning base env, so subsequent
        probe rollouts run against the calibrated sim.
        """
        if approach is None:
            return (f"[{version_tag}] Error: PHYSICAL_PARAMS requires a bound "
                    "approach (raw trajectories + base env) — unavailable in "
                    "this session.")
        # Stamp the fit scale (log vs linear) from the env registry —
        # the agent's declaration carries name/init/bounds only.
        physical_specs = stamp_physical_spec_scales(physical_specs,
                                                    approach._base_env)  # pylint: disable=protected-access
        rollouts = approach._rollout_fit_trajectories(  # pylint: disable=protected-access
            process_features)
        if not rollouts:
            return (f"[{version_tag}] Error: no complete (states, actions) "
                    "trajectories are available, so the rollout system-ID fit "
                    "cannot run. PHYSICAL_PARAMS needs full trajectories, not "
                    "isolated transitions.")
        # Factory, not an instance: every rollout runs in a fresh env.
        fit_env = approach._get_rollout_fit_env()  # pylint: disable=protected-access
        physical_names = [s.name for s in physical_specs]
        init_params = {
            s.name: s.init_value
            for s in list(physical_specs) + list(rule_specs)
        }
        anchors = physical_param_anchors(
            approach._base_env,  # pylint: disable=protected-access
            physical_specs)
        try:
            # One scaling object per fit: every SSE/RMS below must share
            # it or their values are incomparable.
            scaling = compute_residual_scaling(rollouts, process_features)
            pre_sse = compute_rollout_sse(fit_env, rollouts, init_params,
                                          process_features, physical_names,
                                          rules, latent_init, scaling)
            fit_result, survivors, traj_rms = fit_params_rollout_trimmed(
                fit_env,
                rollouts,
                physical_specs,
                process_features,
                rules=rules,
                rule_specs=rule_specs,
                latent_init=latent_init,
                scaling=scaling,
                anchors=anchors,
                rms_cache=getattr(approach, "_explainability_cache", None))
            if not survivors:
                # Honest empty-data output: no fit ran, nothing was
                # applied - do NOT print fitted values or identifiability
                # verdicts computed on zero surviving data (chaos makes
                # the probe report "identified" for everything).
                rms_str = ", ".join(f"{r:.4g}" for r in traj_rms)
                return "\n".join([
                    f"[{version_tag}] NO FIT RAN: all {len(rollouts)} "
                    "recorded motion segments were unexplainable at ANY "
                    "candidate physical parameters (per-segment "
                    f"best-achievable RMS [{rms_str}] all above the "
                    "trimming threshold).",
                    "",
                    "Parameters were left at their baselines; nothing was "
                    "applied to the planning base env.",
                    "",
                    "This usually means the recorded interactions are not "
                    "repeatable under replay (prolonged scraping/jamming "
                    "robot-object contact is chaotic). Collect experiments "
                    "whose outcome is dominated by object dynamics: actuate "
                    "one or two objects cleanly, then let the scene evolve "
                    "and settle on its own.",
                ])
            probe_rollouts = survivors
            fitted = fit_result.point_estimate
            post_sse = compute_rollout_sse(fit_env, probe_rollouts, fitted,
                                           process_features, physical_names,
                                           rules, latent_init, scaling)
        except Exception as e:  # pylint: disable=broad-except
            return (
                f"[{version_tag}] Error: rollout system-ID fit failed:\n{e}")

        def rollout_sse_fn(params: Dict[str, float]) -> float:
            return compute_rollout_sse(fit_env, probe_rollouts, params,
                                       process_features, physical_names, rules,
                                       latent_init, scaling)

        ident_report = identifiability_report(fit_result,
                                              rollout_sse_fn,
                                              list(physical_specs) +
                                              list(rule_specs),
                                              num_explainable=len(survivors))
        applied = select_trustworthy_params(fitted, init_params,
                                            physical_names, ident_report,
                                            anchors)
        approach._apply_identified_physical_params(applied)  # pylint: disable=protected-access
        if hasattr(approach, "_record_sysid_diagnostics"):
            approach._record_sysid_diagnostics(  # pylint: disable=protected-access
                ident_report, physical_names, len(survivors), len(rollouts),
                traj_rms)
        kept_at_init = sorted(n for n in physical_names
                              if applied[n] != fitted[n])
        if pre_sse > 0:
            pct_str = f"({(pre_sse - post_sse) / pre_sse * 100:+.1f}% vs init)"
        else:
            pct_str = "(init SSE was 0)"
        lines = [
            f"[{version_tag}] JOINT ROLLOUT SYSTEM-ID FIT (PHYSICAL_PARAMS "
            f"declared) on {len(rollouts)} motion segments (scope: "
            f"{scope_note}; {len(physical_names)} physical + "
            f"{len(list(rule_specs))} rule params). Residuals are "
            "per-feature normalized (angles wrapped), so SSE/RMS are "
            "dimensionless fractions of typical motion.",
            "",
            f"At init params:   rollout SSE = {pre_sse:.6f}",
            f"After joint fit:  rollout SSE = {post_sse:.6f}  {pct_str}",
            "",
            "Fitted parameters:",
        ]
        if len(survivors) < len(rollouts):
            rms_str = ", ".join(f"{r:.4g}" for r in traj_rms)
            lines.insert(
                1,
                f"Goodness-of-fit trimming: {len(rollouts) - len(survivors)}"
                f" of {len(rollouts)} motion segments were unexplainable at "
                "ANY candidate params (per-segment best-achievable RMS: "
                f"[{rms_str}]) and were dropped before fitting; the fit "
                "below used only the explainable ones. Unexplainable "
                "segments are not repeatable under replay - prefer "
                "experiments whose outcome is dominated by object dynamics "
                "rather than prolonged robot-object contact.")
        for name in sorted(fitted):
            init_val = init_params[name]
            fit_val = fitted[name]
            delta = fit_val - init_val
            ppct = (delta / init_val * 100) if init_val != 0 else float("nan")
            kind = "physical" if name in physical_names else "rule"
            lines.append(f"  {name:<28} [{kind:<8}] {init_val:.4f} -> "
                         f"{fit_val:.4f}  (delta={delta:+.4f}, {ppct:+.1f}%)")

        lines.extend([
            "",
            "Identifiability (posterior_std / prior_std; ~1 means the data "
            "did NOT constrain the parameter — its fitted value is "
            "arbitrary, so remove it from PHYSICAL_PARAMS or collect data "
            "that exercises it):",
            format_identifiability(ident_report),
            "",
        ])
        if kept_at_init:
            lines.append(
                "Applied to the planning base env: fitted values for the "
                "identified params only; "
                f"{', '.join(kept_at_init)} did not contract (or failed "
                "the sensitivity screen), so their baseline values were "
                "kept (the fitted values above for them are arbitrary). "
                "Probe rollouts (sim.run / sim.refine) now run against the "
                "partially calibrated sim.")
        else:
            lines.append(
                "The identified physical params were applied to the "
                "planning base env; probe rollouts (sim.run / sim.refine) "
                "now run against the calibrated sim.")
        return "\n".join(lines)

    # ── run_python ──────────────────────────────────────────

    # The approach merges the ProbeSim facade (`sim` over the CANDIDATE
    # simulator.py) into this same namespace - one exec namespace per
    # synthesis session, so helpers defined next to the data are
    # visible to probe sweeps. Since the fit/refine/forward-validate
    # surfaces all live on `sim` now, the probe is unconditional in
    # synthesis (there is no other validation surface). Shared blurb,
    # so the wording cannot drift from the solve-phase explore_python
    # surface.
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.tools.exploration import probe_api_blurb
    probe_blurb = (
        " This namespace ALSO binds the candidate-simulator probe: " +
        probe_api_blurb(synthesis_probe=True) +
        " Probe rollouts are CANDIDATE-simulator predictions - do not "
        "mix them up with the recorded real `trajectories`. Nothing the "
        "probe runs is captured; the validation protocol before declaring "
        "the simulator done is: `sim.fit()` (canonical fit report), "
        "`sim.refine(plan, require_goal=True)` (params exist that reach "
        "each subgoal), then a continuous `sim.run` of the refined plan "
        "(the forward pass; a refine-pass/run-fail means a rule is more "
        "permissive than the data).")
    run_python = _make_python_exec_tool(
        tool,
        name="run_python",
        description=(
            "Execute Python code for ad-hoc data exploration. Available "
            "variables: trajectories (List[LowLevelTrajectory]; each has "
            "`is_demo`, `train_task_idx`, `states`, `actions`), train_tasks "
            "(List[Task]; each has `init`, `goal`, `goal_holds(state)`), "
            "is_goal_state (callable: state, task_idx -> bool - do the goal "
            "atoms hold in this one STATE; reaching the goal atoms does not "
            "by itself mean solved), describe_trajectory(traj_idx, "
            "include_states=True, include_atoms=False, max_timesteps=10) "
            "- a per-timestep digest of one trajectory, np, ParamSpec, "
            "and (when the "
            "env defines task evaluators) evaluate_trajectory(states, "
            "actions=None, task_idx=0) -> {reward, solved} - the env's "
            "ground-truth episode scoring over a full TRAJECTORY. "
            "print() output "
            "is returned. The namespace persists across calls. If output "
            "exceeds ~30k chars it is saved to "
            "`tool_outputs/run_python/call_NNNN.txt` in the sandbox and only "
            "a head/tail preview plus that path is returned - use Read/Grep "
            "to inspect the full file. This does NOT define rules - write "
            "`simulator.py` for that; `sim.fit` and `sim.residuals` "
            "load PROCESS_RULES, PARAM_SPECS, PROCESS_FEATURES from that "
            "file fresh on every call." + probe_blurb),
        exec_ns=exec_ns,
        sandbox_dir=sandbox_dir,
        sandbox_dir_for_agent=sandbox_dir_for_agent,
        text_result=_text,
    )

    # ── run_fit (the ``sim.fit`` backend) ───────────────────────

    def run_fit(path: Optional[str] = None,
                traj_idxs: Optional[List[int]] = None,
                fixed: Optional[Dict[str, float]] = None) -> str:
        """Fit PARAM_SPECS (loaded fresh from ``simulator.py``) and report.

        The backend behind ``sim.fit``. With no arguments this is the
        CANONICAL fit - the same data/fit the probe deploys - and, when
        the file declares PHYSICAL_PARAMS, the identified physical
        values are applied to the planning base env. Any argument makes
        the fit EXPLORATORY: a diagnostic report only, publishing and
        applying nothing.

        ``traj_idxs`` restricts the fit to those trajectories' step
        transitions; ``fixed`` pins parameters at given values while
        the rest are fit (both rule-param paths only - the rollout
        system-ID path rejects them). Each call snapshots the simulator
        file into simulator_versions/ and tags output
        ``[cycle_XXX_vers_YYY]``.
        """
        p = path or simulator_file
        rules, specs, declared, latent_init, physical_specs, version_tag, \
            err = _snapshot_and_load(p)
        if err:
            return str(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)
        scope_note = ("declared" if isinstance(declared, dict) else
                      "inferred (PROCESS_FEATURES not declared)")
        canonical = traj_idxs is None and not fixed

        # PHYSICAL_PARAMS declared -> joint system-identification fit on
        # free-running rollouts (the per-transition/teacher-forced paths
        # below cannot see physical params: State carries no velocities).
        if physical_specs:
            if not canonical:
                return (f"[{version_tag}] Error: traj_idxs/fixed are not "
                        "supported with PHYSICAL_PARAMS (the rollout "
                        "system-ID fit trims and weights motion segments "
                        "itself). Pin a physical param by narrowing its "
                        "lo/hi bounds in PHYSICAL_PARAMS instead.")
            return _evaluate_rollout_fit(rules, specs, physical_specs,
                                         latent_init, process_features,
                                         scope_note, version_tag)

        if fixed:
            known = {s.name for s in specs}
            unknown = sorted(set(fixed) - known)
            if unknown:
                return (f"[{version_tag}] Error: fixed names {unknown} are "
                        f"not in PARAM_SPECS (available: {sorted(known)}).")

        triples = base_pred_triples
        groups = _groups_for(base_pred_triples)
        if traj_idxs is not None:
            bad = sorted(i for i in traj_idxs if not 0 <= i < len(groups))
            if bad:
                return (f"[{version_tag}] Error: traj_idxs {bad} out of "
                        f"range (0-{len(groups) - 1}).")
            groups = [groups[i] for i in traj_idxs]
            triples = [t for g in groups for t in g]
            if not triples:
                return (f"[{version_tag}] Error: the selected trajectories "
                        "contain no step transitions.")

        # Dispatch on the rule signature exactly as the fitting engine
        # does: recurrent (5-arg, latent-declaring) rules are scored with
        # the latent block threaded per trajectory, never through the
        # legacy per-transition path (which would call them with 3 args).
        latent_mode = has_latent_rules(rules)
        # Pinning: wrap the rules so pinned values override whatever the
        # fit engine proposes, and fit only the free specs. Wrapper arg
        # names must be preserved - the engine dispatches recurrent
        # rules by inspecting for a 2nd parameter named ``latent``.
        fit_rules = rules
        fit_specs = list(specs)
        if fixed:
            fixed_vals = dict(fixed)
            if latent_mode:
                fit_rules = [
                    lambda state, latent, history, updates, params, _r=r: _r(
                        state, latent, history, updates, {
                            **params,
                            **fixed_vals
                        }) for r in rules
                ]
            else:
                fit_rules = [
                    lambda state, updates, params, _r=r: _r(
                        state, updates, {
                            **params,
                            **fixed_vals
                        }) for r in rules
                ]
            fit_specs = [s for s in specs if s.name not in fixed_vals]
        init_params = {s.name: s.init_value for s in fit_specs}
        try:
            if latent_mode:
                pre_sse = compute_sse_recurrent(fit_rules, groups, init_params,
                                                latent_init, process_features)
            else:
                sim_fn = lambda s, _a, prm: apply_rules(  # noqa: E731
                    s, fit_rules, prm)
                pre_sse = compute_sse(sim_fn, triples, init_params,
                                      process_features)
        except Exception as e:  # pylint: disable=broad-except
            return f"[{version_tag}] Error: SSE computation failed:\n{e}"

        sig_note = ("recurrent (latent threaded per trajectory)"
                    if latent_mode else "per-transition")
        mode_note = ("canonical - the same fit the probe deploys" if canonical
                     else "EXPLORATORY - diagnostic only, nothing published")
        lines = [
            f"[{version_tag}] Fit evaluation on {len(triples)} "
            f"step transitions (scope: {scope_note}; rules: {sig_note}; "
            f"{mode_note}).",
        ]
        if traj_idxs is not None:
            lines.append(f"Restricted to trajectories {sorted(traj_idxs)}.")
        if fixed:
            pinned = ", ".join(f"{n}={v:.4g}"
                               for n, v in sorted(fixed.items()))
            lines.append(f"Pinned parameters: {pinned}.")
        lines += ["", f"At init_value params:  SSE = {pre_sse:.6f}"]

        if not fit_specs:
            lines.append(
                "All parameters are pinned - no fit ran; the SSE above is "
                "the score at the pinned values.")
            return "\n".join(lines)
        try:
            if latent_mode:
                fit_result, post_sse = fit_rule_parameters_latent(
                    fit_rules, fit_specs, groups, latent_init,
                    process_features)
            else:
                fit_result, post_sse = fit_rule_parameters(
                    fit_rules, fit_specs, triples, process_features)
            fitted_params = fit_result.point_estimate
        except Exception as e:  # pylint: disable=broad-except
            return f"[{version_tag}] Error: fit_params failed:\n{e}"
        if pre_sse > 0:
            pct = (pre_sse - post_sse) / pre_sse * 100
            pct_str = f"({pct:+.1f}% vs init)"
        else:
            pct_str = "(init SSE was 0)"
        lines.append(f"After fit:             SSE = {post_sse:.6f}  "
                     f"{pct_str}")
        lines.append("")
        lines.append("Fitted parameters:")
        for name in sorted(fitted_params):
            init_val = init_params[name]
            fit_val = fitted_params[name]
            delta = fit_val - init_val
            ppct = ((delta / init_val *
                     100) if init_val != 0 else float("nan"))
            lines.append(f"  {name:<30} {init_val:.4f} -> "
                         f"{fit_val:.4f}  (delta={delta:+.4f}, "
                         f"{ppct:+.1f}%)")

        return "\n".join(lines)

    # ── run_residuals (the ``sim.residuals`` backend) ───────────

    def run_residuals(max_transitions: int = 100,
                      abs_tol: float = 1e-4,
                      rel_tol: float = 1e-3,
                      num_worst_examples: int = 3,
                      fit_params: bool = False,
                      path: Optional[str] = None) -> str:
        """Per-feature residual report for the current PROCESS_RULES.

        The backend behind ``sim.residuals``. Loads the rules fresh
        from ``simulator.py`` and reports, for each feature in
        PROCESS_FEATURES (or the inferred fallback), the mismatch
        count, mean/max abs error, and the relative improvement over
        the no-rule baseline (negative means the rules are worse than
        not running them at all), plus the worst-N example transitions
        per feature. Uses init_value params by default;
        ``fit_params=True`` MCMC-fits first (diagnostic only - nothing
        is published). Tolerance: ``|pred - obs| > rel_tol * |obs| +
        abs_tol``. Each call snapshots the simulator file into
        simulator_versions/ and tags output ``[cycle_XXX_vers_YYY]``.
        """
        p = path or simulator_file
        rules, specs, declared, latent_init, _physical_specs, version_tag, \
            err = _snapshot_and_load(p)
        if err:
            return str(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)
        scope_label = ("declared"
                       if isinstance(declared, dict) else "inferred")

        max_n = int(max_transitions)
        abs_tol = float(abs_tol)
        rel_tol = float(rel_tol)
        n_examples = int(num_worst_examples)
        do_fit = bool(fit_params)

        # Same engine-matching dispatch as run_fit: recurrent
        # rules are fit and rolled out with the latent threaded per
        # trajectory, never called per-transition with 3 args.
        latent_mode = has_latent_rules(rules)
        groups = _groups_for(base_pred_triples)
        if do_fit:
            try:
                if latent_mode:
                    fit_result, _ = fit_rule_parameters_latent(
                        rules, specs, groups, latent_init, process_features)
                else:
                    fit_result, _ = fit_rule_parameters(
                        rules, specs, base_pred_triples, process_features)
                t_params = fit_result.point_estimate
                param_label = "fitted"
            except Exception as e:  # pylint: disable=broad-except
                return f"[{version_tag}] Error: param fitting failed:\n{e}"
        else:
            t_params = {s.name: s.init_value for s in specs}
            param_label = "init_value"

        # Predicted next states, latent threaded per trajectory for
        # recurrent rules (legacy rules roll each transition independently).
        # Roll out all groups in flat order, then truncate to max_n so the
        # reported step indices line up with the flat triples slice below.
        try:
            all_preds = rollout_predictions(rules, t_params, groups,
                                            latent_init)
        except Exception as e:  # pylint: disable=broad-except
            return f"[{version_tag}] Error: rule rollout failed:\n{e}"
        triples_rules: List = all_preds[:max_n]
        triples_base: List = [(bs, sn)
                              for bs, _a, sn in base_pred_triples[:max_n]]

        # Per-feature accumulators keyed by (type_name, feat_name).
        rule_n_total: Dict = defaultdict(int)
        rule_n_mismatch: Dict = defaultdict(int)
        rule_sum_err: Dict = defaultdict(float)
        rule_max_err: Dict = defaultdict(float)
        base_n_total: Dict = defaultdict(int)
        base_sum_err: Dict = defaultdict(float)
        worst: Dict = defaultdict(list)
        mismatched_steps: set = set()

        for i, obj, tn, feat, pred, obs in iter_feature_residuals(
                triples_rules, process_features):
            key = (tn, feat)
            err = abs(pred - obs)
            thr = rel_tol * abs(obs) + abs_tol
            rule_n_total[key] += 1
            rule_sum_err[key] += err
            if err > rule_max_err[key]:
                rule_max_err[key] = err
            if err > thr:
                rule_n_mismatch[key] += 1
                mismatched_steps.add(i)
                worst[key].append((i, obj.name, pred, obs, err))

        for _, _, tn, feat, pred, obs in iter_feature_residuals(
                triples_base, process_features):
            key = (tn, feat)
            base_n_total[key] += 1
            base_sum_err[key] += abs(pred - obs)

        if not rule_n_total:
            return (f"[{version_tag}] PROCESS_FEATURES is empty; "
                    "nothing to report.")

        n_steps = len(triples_rules)
        perfect_steps = n_steps - len(mismatched_steps)
        lines = [
            f"[{version_tag}] Residual report — {n_steps} step transitions, "
            f"scope: {scope_label} PROCESS_FEATURES, "
            f"params: {param_label}, "
            f"tol: {rel_tol:g}*|obs| + {abs_tol:g}.",
            f"Steps with all in-scope features within tol: "
            f"{perfect_steps}/{n_steps}.",
            "",
            f"{'feature':<35} {'misses/total':<14} {'mean_err':<10} "
            f"{'max_err':<10} {'vs base':<14}",
        ]
        for key in sorted(rule_n_total):
            tn, feat = key
            n_tot = rule_n_total[key]
            n_mm = rule_n_mismatch[key]
            mean = rule_sum_err[key] / max(1, n_tot)
            mx = rule_max_err[key]
            bn = max(1, base_n_total[key])
            base_mean = base_sum_err[key] / bn
            if base_mean > 0:
                improvement = (base_mean - mean) / base_mean * 100
                vs_base = f"{improvement:+.0f}%"
                if improvement < 0:
                    vs_base += " (worse)"
            elif mean == 0:
                vs_base = "exact"
            else:
                vs_base = "rules add err"
            lines.append(f"{tn + '.' + feat:<35} {f'{n_mm}/{n_tot}':<14} "
                         f"{mean:<10.4f} {mx:<10.4f} {vs_base:<14}")

        if n_examples > 0 and worst:
            lines.append("")
            lines.append(f"Worst {n_examples} mismatches per feature "
                         f"(step N = trajectory transition state[N] -> "
                         f"state[N+1]):")
            for key in sorted(worst):
                tn, feat = key
                entries = sorted(worst[key], key=lambda x: x[4], reverse=True)
                for step, oname, pred, obs, err in entries[:n_examples]:
                    lines.append(f"  step {step:>4}  {oname}.{feat}: "
                                 f"pred={pred:.6f} obs={obs:.6f} "
                                 f"err={err:.6f}")

        return "\n".join(lines)

    return SynthesisToolkit(tools=[run_python],
                            fit_runner=run_fit,
                            residuals_runner=run_residuals)
