"""Synthesis-session tools for sim learning (create_synthesis_tools)."""
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators.agent_sdk.synthesis_backend import SynthesisBackend
from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths
from predicators.agent_sdk.tools.snapshots import _ArtifactSnapshotter


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
) -> list:
    """Create MCP tools for the sim-learning synthesis agent.

    Returns ``[run_python, evaluate_step_fit, report_residuals,
    evaluate_plan_refinement]``.

    The agent's source-of-truth for the simulator is the file at
    ``simulator_file`` (which it edits with ``Write`` / ``Edit``). The
    three synthesis tools each ``exec`` that file fresh into an
    isolated namespace per call and read ``PROCESS_RULES``,
    ``PARAM_SPECS``, ``PROCESS_FEATURES`` from it — no namespace state
    leaks across iterations. Before loading, every call also snapshots
    the current contents into ``versions_dir`` as
    ``cycle_XXX_vers_YYY_simulator.py`` (``XXX`` from
    ``cycle_index_provider()``, ``YYY`` resetting per
    ``create_synthesis_tools`` call) so the full history of evaluated
    versions is preserved across cycles; identical-content calls reuse
    the prior snapshot. Each tool's output is prefixed with the version
    tag (``[cycle_XXX_vers_YYY]``).

    * ``run_python`` — executes arbitrary Python in a persistent
      namespace pre-loaded with trajectory data. Use this for ad-hoc
      exploration of ``trajectories`` etc.; it does **not** define
      rules — write ``simulator.py`` for that.
    * ``evaluate_step_fit`` — SSE of the current ``PROCESS_RULES`` at
      init_value params, plus post-fit SSE, percent improvement, and
      fitted parameter values from a parameter fit.
    * ``report_residuals`` — per-feature breakdown of where the
      current rules disagree with observations: mismatch counts,
      mean/max abs error, comparison to the no-rule baseline, and
      worst-N example transitions per feature.
    * ``evaluate_plan_refinement`` — builds the combined simulator
      from current rules+params and runs backtracking refinement on a
      training task, reporting where (if anywhere) the planner gets
      stuck. Requires ``approach`` to be passed.

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
        approach: ``AgentSimLearningApproach`` instance, used by
            ``evaluate_plan_refinement`` to access training tasks,
            build the combined simulator/option model, and run
            refinement. If ``None``, that tool returns an error.
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

    from predicators.approaches.synthesis_validation import \
        run_refinement_for_synthesis
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
                              scope_note: str,
                              version_tag: str) -> Dict[str, Any]:
        """Joint physical+rule system-ID fit on free-running rollouts.

        Reached from ``evaluate_step_fit`` when the artifact declares
        ``PHYSICAL_PARAMS``. Needs the bound approach for the raw
        (states, actions) trajectories and the dedicated headless fit
        env. On success the identified physical values are applied in
        place to the approach's planning base env, so a subsequent
        ``evaluate_plan_refinement`` call plans against the calibrated
        sim.
        """
        if approach is None:
            return _text(
                f"[{version_tag}] Error: PHYSICAL_PARAMS requires a bound "
                "approach (raw trajectories + base env) — unavailable in "
                "this session.")
        # Stamp the fit scale (log vs linear) from the env registry —
        # the agent's declaration carries name/init/bounds only.
        physical_specs = stamp_physical_spec_scales(physical_specs,
                                                    approach._base_env)  # pylint: disable=protected-access
        rollouts = approach._rollout_fit_trajectories(  # pylint: disable=protected-access
            process_features)
        if not rollouts:
            return _text(
                f"[{version_tag}] Error: no complete (states, actions) "
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
                return _text("\n".join([
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
                ]))
            probe_rollouts = survivors
            fitted = fit_result.point_estimate
            post_sse = compute_rollout_sse(fit_env, probe_rollouts, fitted,
                                           process_features, physical_names,
                                           rules, latent_init, scaling)
        except Exception as e:  # pylint: disable=broad-except
            return _text(
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
                "evaluate_plan_refinement now plans against the partially "
                "calibrated sim.")
        else:
            lines.append(
                "The identified physical params were applied to the "
                "planning base env; evaluate_plan_refinement now plans "
                "against the calibrated sim.")
        return _text("\n".join(lines))

    # ── run_python ──────────────────────────────────────────

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
            "by itself mean solved), np, ParamSpec, and (when the "
            "env defines task evaluators) evaluate_trajectory(states, "
            "actions=None, task_idx=0) -> {reward, solved} - the env's "
            "ground-truth episode scoring over a full TRAJECTORY. "
            "print() output "
            "is returned. The namespace persists across calls. If output "
            "exceeds ~30k chars it is saved to "
            "`tool_outputs/run_python/call_NNNN.txt` in the sandbox and only "
            "a head/tail preview plus that path is returned - use Read/Grep "
            "to inspect the full file. This does NOT define rules - write "
            "`simulator.py` for that; the synthesis tools "
            "(evaluate_step_fit, report_residuals, evaluate_plan_refinement) "
            "load PROCESS_RULES, PARAM_SPECS, PROCESS_FEATURES from that "
            "file."),
        exec_ns=exec_ns,
        sandbox_dir=sandbox_dir,
        sandbox_dir_for_agent=sandbox_dir_for_agent,
        text_result=_text,
    )

    # ── evaluate_step_fit ────────────────────────────────────────

    @tool(
        "evaluate_step_fit",
        "Score the current PROCESS_RULES (loaded fresh from "
        "`simulator.py`) by SSE on the step transitions. Reports SSE "
        "at init_value params from PARAM_SPECS, then fits parameters "
        "and reports the post-fit SSE plus percent improvement and the "
        "fitted parameter values with their delta from init. If the "
        "file declares PHYSICAL_PARAMS (base-sim system "
        "identification), the fit instead matches free-running "
        "base-sim rollouts of full trajectories, fits physical + rule "
        "params jointly, reports per-parameter identifiability, and "
        "applies the identified physical values to the planning base "
        "env. Each call snapshots the simulator file into "
        "simulator_versions/; output is tagged [cycle_XXX_vers_YYY].",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type":
                    "string",
                    "description":
                    "Override simulator file path "
                    "(defaults to the canonical simulator.py).",
                },
            },
        },
    )
    async def evaluate_step_fit(args: Dict[str, Any]) -> Dict[str, Any]:
        path = args.get("path") or simulator_file
        rules, specs, declared, latent_init, physical_specs, version_tag, \
            err = _snapshot_and_load(path)
        if err:
            return _text(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)
        scope_note = ("declared" if isinstance(declared, dict) else
                      "inferred (PROCESS_FEATURES not declared)")

        # PHYSICAL_PARAMS declared -> joint system-identification fit on
        # free-running rollouts (the per-transition/teacher-forced paths
        # below cannot see physical params: State carries no velocities).
        if physical_specs:
            return _evaluate_rollout_fit(rules, specs, physical_specs,
                                         latent_init, process_features,
                                         scope_note, version_tag)

        # Dispatch on the rule signature exactly as the fitting engine
        # does: recurrent (5-arg, latent-declaring) rules are scored with
        # the latent block threaded per trajectory, never through the
        # legacy per-transition path (which would call them with 3 args).
        latent_mode = has_latent_rules(rules)
        init_params = {s.name: s.init_value for s in specs}
        try:
            if latent_mode:
                groups = _groups_for(base_pred_triples)
                pre_sse = compute_sse_recurrent(rules, groups, init_params,
                                                latent_init, process_features)
            else:
                sim_fn = lambda s, _a, p: apply_rules(  # noqa: E731
                    s, rules, p)
                pre_sse = compute_sse(sim_fn, base_pred_triples, init_params,
                                      process_features)
        except Exception as e:  # pylint: disable=broad-except
            return _text(
                f"[{version_tag}] Error: SSE computation failed:\n{e}")

        sig_note = ("recurrent (latent threaded per trajectory)"
                    if latent_mode else "per-transition")
        lines = [
            f"[{version_tag}] Fit evaluation on {len(base_pred_triples)} "
            f"step transitions (scope: {scope_note}; rules: {sig_note}).",
            "",
            f"At init_value params:  SSE = {pre_sse:.6f}",
        ]

        try:
            if latent_mode:
                fit_result, post_sse = fit_rule_parameters_latent(
                    rules, specs, groups, latent_init, process_features)
            else:
                fit_result, post_sse = fit_rule_parameters(
                    rules, specs, base_pred_triples, process_features)
            fitted_params = fit_result.point_estimate
        except Exception as e:  # pylint: disable=broad-except
            return _text(f"[{version_tag}] Error: fit_params failed:\n{e}")
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

        return _text("\n".join(lines))

    # ── report_residuals ────────────────────────────────────

    @tool(
        "report_residuals",
        "Per-feature breakdown of where the current PROCESS_RULES "
        "(loaded fresh from `simulator.py`) disagree with "
        "observations on step transitions. For each feature in "
        "PROCESS_FEATURES (or the inferred fallback) reports mismatch "
        "count, mean abs error, max abs error, and the relative "
        "improvement over the no-rule baseline (negative means rules "
        "are worse than not running them at all). Also lists the "
        "worst-N example transitions per feature so you can see what "
        "edge cases break. Uses init_value from PARAM_SPECS by "
        "default; pass fit_params=true to MCMC-fit first. Tolerance: "
        "|pred - obs| > rel_tol * |obs| + abs_tol. Each call "
        "snapshots the simulator file into simulator_versions/; "
        "output is tagged [cycle_XXX_vers_YYY].",
        {
            "type": "object",
            "properties": {
                "max_transitions": {
                    "type": "integer",
                    "description": "Max transitions to inspect "
                    "(default 100).",
                },
                "abs_tol": {
                    "type": "number",
                    "description": "Absolute tolerance (default 1e-4).",
                },
                "rel_tol": {
                    "type": "number",
                    "description": "Relative tolerance (default 1e-3).",
                },
                "num_worst_examples": {
                    "type":
                    "integer",
                    "description":
                    "Worst-N mismatched transitions to "
                    "list per feature (default 3, 0 to suppress).",
                },
                "fit_params": {
                    "type":
                    "boolean",
                    "description":
                    "If true, run MCMC fit before "
                    "computing residuals; otherwise use init_value "
                    "(default false).",
                },
                "path": {
                    "type":
                    "string",
                    "description":
                    "Override simulator file path "
                    "(defaults to the canonical simulator.py).",
                },
            },
        },
    )
    async def report_residuals(args: Dict[str, Any]) -> Dict[str, Any]:
        path = args.get("path") or simulator_file
        rules, specs, declared, latent_init, _physical_specs, version_tag, \
            err = _snapshot_and_load(path)
        if err:
            return _text(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)
        scope_label = ("declared"
                       if isinstance(declared, dict) else "inferred")

        max_n = int(args.get("max_transitions", 100))
        abs_tol = float(args.get("abs_tol", 1e-4))
        rel_tol = float(args.get("rel_tol", 1e-3))
        n_examples = int(args.get("num_worst_examples", 3))
        do_fit = bool(args.get("fit_params", False))

        # Same engine-matching dispatch as evaluate_step_fit: recurrent
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
                return _text(
                    f"[{version_tag}] Error: param fitting failed:\n{e}")
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
            return _text(f"[{version_tag}] Error: rule rollout failed:\n{e}")
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
            return _text(f"[{version_tag}] PROCESS_FEATURES is empty; "
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

        return _text("\n".join(lines))

    # ── evaluate_plan_refinement ────────────────────────────

    @tool(
        "evaluate_plan_refinement",
        "MCMC-fit PARAM_SPECS (loaded fresh from `simulator.py`), "
        "build the combined simulator from current PROCESS_RULES + "
        "the fitted params, then run **both** backtracking refinement "
        "and continuous forward validation on a training task against "
        "a plan you propose. Always fits first because refinement "
        "needs to test the simulator at its deployed (fitted) params, "
        "not at init_value. `plan` is required — pass the "
        "option-skeleton you believe should solve the task, one "
        "option call per line, with every option argument supplied "
        "and typed object references (`obj:type`) matching what the "
        "inspect tools report. The parser is strict and will not "
        "auto-fill omitted arguments. Example shape (substitute the "
        "options/types/predicates your task actually exposes): "
        "`PickWidget(robot:robot, widget0:widget)\\nPlace(robot:robot) "
        "-> {WidgetAtFixture(widget0:widget, fixture0:fixture)}\\n...`. "
        "Subgoal annotations (`-> {Atom(obj:type, ...)}`) are "
        "optional in general but effectively required after "
        "open-ended skills like `Place`: without a subgoal the "
        "search has no preference for *where* to put the object, so "
        "a downstream `Wait` may get stuck and look like a rule bug. "
        "For `Wait`, the annotation also specifies when the wait "
        "should terminate; prefix an atom with `NOT` to require it "
        "become false. The `timeout` argument auto-scales with "
        "sketch length when omitted (see the `timeout` field "
        "below). Reports the verdict for refinement (success, "
        "TIMEOUT, SAMPLE_EXHAUSTED with stuck step) and — when "
        "refinement passes — also the verdict for forward validation "
        "(SUCCESS, or FORWARD_VALIDATION_FAILED with the first "
        "subgoal/goal divergence). Refinement may pass while forward "
        "validation fails: refinement resets state between options "
        "and resamples up to 50× per step, while forward validation "
        "runs the same plan once continuously. A refinement-pass "
        "+ forward-validation-fail almost always means a learned "
        "threshold/rule is more permissive than the env's effective "
        "behavior, so refinement believes a subgoal holds when the "
        "env-driven post-state actually doesn't. The agent must "
        "treat forward-validation failure the same as refinement "
        "failure — keep iterating, do not declare done. Each call "
        "snapshots the simulator file into simulator_versions/; "
        "output is tagged [cycle_XXX_vers_YYY]. Slow — use sparingly.",
        {
            "type": "object",
            "properties": {
                "plan": {
                    "type":
                    "string",
                    "description":
                    "Option-skeleton plan text, one "
                    "option call per line. Use typed object "
                    "references (`obj:type`) and supply every "
                    "option argument. Optional `-> {Atom(...)}` "
                    "subgoal after each step; effectively required "
                    "after open-ended skills like `Place`.",
                },
                "task_idx": {
                    "type": "integer",
                    "description": "Index into training tasks "
                    "(default 0).",
                },
                "timeout": {
                    "type":
                    "number",
                    "description":
                    "Refinement timeout in seconds. Omit "
                    "for an auto value that scales with the "
                    "number of steps in the sketch; the actual "
                    "value used is reported back. Override only "
                    "if the previous report said TIMEOUT. MCMC "
                    "fitting runs before refinement and is not "
                    "subject to this timeout.",
                },
                "path": {
                    "type":
                    "string",
                    "description":
                    "Override simulator file path "
                    "(defaults to the canonical simulator.py).",
                },
            },
        },
    )
    async def evaluate_plan_refinement(args: Dict[str, Any]) -> Dict[str, Any]:
        if approach is None:
            return _text("Error: evaluate_plan_refinement is unavailable "
                         "(no approach instance bound to the tool).")

        path = args.get("path") or simulator_file
        rules, specs, declared, latent_init, _physical_specs, version_tag, \
            err = _snapshot_and_load(path)
        if err:
            return _text(err)

        process_features = (declared if isinstance(declared, dict) else
                            inferred_process_features)

        task_idx = int(args.get("task_idx", 0))
        # Treat missing/None timeout as "auto-scale by sketch length"
        # (computed inside run_refinement_for_synthesis from
        # agent_bilevel_refinement_timeout_per_step / _min).
        timeout_arg = args.get("timeout", None)
        timeout = float(timeout_arg) if timeout_arg is not None else None
        plan_text = args.get("plan", "") or ""

        try:
            report = run_refinement_for_synthesis(
                approach,
                rules=rules,
                specs=specs,
                process_features=process_features,
                base_pred_triples=base_pred_triples,
                task_idx=task_idx,
                timeout=timeout,
                plan_text=plan_text,
                latent_init=latent_init,
            )
        except Exception:  # pylint: disable=broad-except
            tb = _scrub_host_paths(traceback.format_exc())
            return _text(f"[{version_tag}] Error: validation failed:\n{tb}")

        return _text(f"[{version_tag}] {report}")

    return [
        report_residuals,
        run_python,
        evaluate_step_fit,
        evaluate_plan_refinement,
    ]
