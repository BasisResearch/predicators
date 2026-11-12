"""The ``sim.fit`` report of EMPIRIC from assets at one value per parameter.

Under ``continual_point_values_only`` the report of a rollout fit gives
the residual error at the declared and at the fitted values, each fitted
value, and the parameters the recordings do not constrain, in plain
words. It reads the same fit outcome as the other arms' report and names
no belief, no draw and no noise.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from predicators.code_sim_learning.evidence import LaplaceEvidence
from predicators.code_sim_learning.identifiability import Verdict
from predicators.settings import CFG

# What the recordings say about a fitted value, per verdict of the fit.
_CONSTRAINT = {
    Verdict.IDENTIFIED:
    "constrained by the recordings",
    Verdict.WEAKLY_IDENTIFIED:
    "weakly constrained by the recordings",
    Verdict.WIDE:
    "weakly constrained by the recordings",
    Verdict.NOT_IDENTIFIED:
    "NOT constrained by the recordings: other values explain them as "
    "well, so the fitted value is arbitrary",
    Verdict.INSENSITIVE:
    "NOT constrained by the recordings: no value in its declared range "
    "changes a scored feature",
    Verdict.AT_BOUND:
    "fitted to an end of its declared range: the recordings push it out "
    "of the range, which often means a missing mechanism",
    Verdict.ANCHORED:
    "NOT constrained by the recordings: the declared value explains them "
    "as well as the fitted one",
    Verdict.INCONSISTENT:
    "NOT settled: this fit's value is far from the last fit's",
    Verdict.UNKNOWN:
    "could not be assessed on these recordings",
}


def search_tried() -> str:
    """The parameter values a fit tries before it leaves a segment out."""
    tried = ("the declared values and each parameter varied alone across "
             "its declared range")
    points = int(CFG.code_sim_learning_rollout_global_design_points)
    if CFG.code_sim_learning_prior_spans_bounds and points > 0:
        tried += f", and {points} points placed over all the ranges together"
    return tried


def left_out_cause(traj_rms: Sequence[float], limit: float) -> List[str]:
    """What to do about the segments a fit left out."""
    over = [r for r in traj_rms if r > limit]
    if not over:
        return []
    return [
        f"{len(over)} recorded segment(s) stayed above the limit "
        f"{limit:.4g} at every value tried (best RMS error "
        f"{min(over):.4g}-{max(over):.4g}). This alone does not name the "
        "cause: an error in the model's structure, an inexact start state, "
        "a search that missed the right values, and contact-sensitive "
        "replay can all contribute. Use sim.validate() on the existing "
        "recordings, compare dynamics structures, and test repeatability "
        "in simulation before spending more real steps. Collect a new "
        "recording only when its predicted outcomes would change the next "
        "action."
    ]


def departure_lines(lines: Sequence[str]) -> List[str]:
    """The lines on where the model leaves each segment a fit left out
    (``describe_departures``), in this report's words."""
    return [
        line.replace("best values from that sweep",
                     "best values from that search") for line in lines
    ]


def evidence_lines(
        evidence: Optional[LaplaceEvidence],
        previous: Optional[Mapping[str, LaplaceEvidence]]) -> List[str]:
    """The report's lines on a fit's model evidence, the score that compares
    two versions of a model on the same recordings."""
    if evidence is None:
        return ["Model evidence: not available for this fit."]
    line = (f"Model evidence (log scale, higher is better): "
            f"{evidence.log_evidence:.1f} on {evidence.num_residuals} "
            f"residuals with {evidence.num_params} parameter(s); the term "
            f"for its parameters is {evidence.log_occam:.1f}")
    if previous:
        (tag, prev), = previous.items()
        if evidence.comparable_with(prev):
            delta = evidence.log_evidence - prev.log_evidence
            winner = ("this version wins" if delta > 0 else
                      "the previous version wins" if delta < 0 else "a tie")
            line += (f". Against {tag}: {prev.log_evidence:.1f}, change "
                     f"{delta:+.1f} ({winner} on evidence)")
        else:
            line += (f". {tag} scored {prev.log_evidence:.1f} on "
                     f"{prev.num_residuals} residuals: a different residual "
                     "set (scope or segments used), so the two are not "
                     "comparable")
    return [
        line + ".",
        "A version that adds parameters has to win on evidence, not on "
        "residual error: the evidence charges for every parameter, so a "
        "parameter the recordings do not need lowers it even as the SSE "
        "falls.",
    ]


def _unscored_line(unscored: Mapping[str, Sequence[str]]) -> str:
    listed = "; ".join(f"{type_name}: {', '.join(features)}"
                       for type_name, features in unscored.items())
    return ("Not scored by this fit, though they change in the recordings: "
            f"{listed}. The fit never compares your model's predictions of "
            "these with the recordings, so a plan that relies on them "
            "(where an object is when it could hit something, what a goal "
            "or failure condition reads) relies on unchecked predictions: "
            "score them in RESIDUAL_FEATURES, or compare them with the "
            "recordings yourself.")


def no_fit_report(*, version_tag: str, scope: str, traj_rms: Sequence[float],
                  limit: float, exploratory: bool,
                  departures: Sequence[str]) -> str:
    """The report of a fit that could explain no recorded segment."""
    rms = ", ".join(f"{r:.4g}" for r in traj_rms)
    kept = ("Exploratory call: nothing recorded." if exploratory else
            "This result is recorded as the fit of this file. An earlier "
            "fit of the same file is kept if there is one; otherwise the "
            "model is UNVALIDATED at its declared values.")
    return "\n".join([
        f"[{version_tag}] NO FIT RAN: none of the {len(traj_rms)} recorded "
        "motion segments could be explained at any parameter values tried "
        f"({search_tried()}); their best RMS errors [{rms}] are all above "
        f"the limit {limit:.4g}. Scored features: {scope}.",
        "",
        "The parameters keep their declared values, and nothing was "
        "applied to the model that sim.run and sim.refine roll out.",
        kept,
        "",
        *left_out_cause(traj_rms, limit),
        *departure_lines(departures),
    ])


def _parameter_lines(name: str, declared: float, fitted: float, runs: float,
                     info: Mapping[str, Any], num_used: int) -> List[str]:
    """One parameter's line: declared -> fitted, what the recordings say about
    it, and the value the model runs when that is not the fitted one."""
    delta = fitted - declared
    pct = (delta / declared * 100) if declared != 0 else float("nan")
    verdict = info.get("verdict", Verdict.UNKNOWN)
    says = _CONSTRAINT.get(verdict, _CONSTRAINT[Verdict.UNKNOWN])
    if verdict is Verdict.AT_BOUND and info.get("bound_edge"):
        says = says.replace("an end", f"the {info['bound_edge']} end")
    if verdict is Verdict.WEAKLY_IDENTIFIED and num_used < 2:
        says += f" (only {num_used} recorded segment(s) support it)"
    if not np.isclose(runs, fitted, rtol=1e-12, atol=0.0):
        says += f"; the model runs it at {runs:.4g}"
    lines = [
        f"  {name:<28} {declared:.4f} -> {fitted:.4f}  "
        f"({delta:+.4f}, {pct:+.1f}%)  {says}"
    ]
    flat = info.get("flat_interval")
    if flat is not None and flat[0] != flat[1]:
        lines.append("      the recordings are explained equally well from "
                     f"{flat[0]:.4g} to {flat[1]:.4g}")
    return lines


def fit_report(*, version_tag: str, exploratory: bool,
               traj_idxs: Optional[Sequence[int]], scope: str,
               num_segments: int, num_used: int, sse_declared: float,
               sse_declared_used: float, sse_fitted: float,
               declared: Mapping[str, float], fitted: Mapping[str, float],
               runs: Mapping[str, float], constraints: Mapping[str,
                                                               Mapping[str,
                                                                       Any]],
               traj_rms: Sequence[float], limit: float,
               unscored: Mapping[str, Sequence[str]], evidence: Sequence[str],
               departures: Sequence[str]) -> str:
    """The report of a rollout fit.

    ``declared``, ``fitted`` and ``runs`` map each parameter to its
    declared value, its fitted value and the value the model runs after
    this fit; ``constraints`` is the fit's per-parameter verdicts
    (``identifiability_report``). ``sse_declared`` is over all
    ``num_segments`` segments and ``sse_declared_used`` and
    ``sse_fitted`` over the ``num_used`` the fit kept. ``evidence`` and
    ``departures`` are lines already in this report's words.
    """
    mode = (f"EXPLORATORY, trajectories {sorted(traj_idxs or [])} only"
            if exploratory else "canonical")
    part = num_used < num_segments
    used = f" on the {num_used} segments the fit used" if part else ""
    if sse_declared_used > 0:
        change = (sse_declared_used - sse_fitted) / sse_declared_used * 100
        lower = f"({change:.1f}% lower than at the declared values" + (
            ", on the same segments)" if part else ")")
    else:
        lower = "(the error at the declared values was 0)"
    lines = [
        f"[{version_tag}] FIT ({mode}) of {len(fitted)} declared "
        f"parameter(s) on {num_segments} recorded motion segments. Scored "
        f"features: {scope}.",
    ]
    if unscored:
        lines.append(_unscored_line(unscored))
    if part:
        rms = ", ".join(f"{r:.4g}" for r in traj_rms)
        lines.append(
            f"Segments left out: {num_segments - num_used} of "
            f"{num_segments} recorded motion segments could not be "
            f"explained at any parameter values tried ({search_tried()}); "
            f"the best RMS error per segment was [{rms}] against the limit "
            f"{limit:.4g}. The fit used the others. " +
            " ".join(left_out_cause(traj_rms, limit)))
        lines.extend(departure_lines(departures))
    if fitted and all(
            np.isclose(fitted[n], declared[n], rtol=1e-8, atol=1e-12)
            for n in fitted):
        lines.append(
            "NOTE: the fit moved no parameter. Every value below is the "
            "declared init_value, not an estimate: the recordings the fit "
            "used do not pull any parameter away from where it started.")
    lines += [
        "",
        "Residual error (SSE: the sum of squared differences between each "
        "replay and its recording, each feature scaled by its typical "
        "motion and angles wrapped, so it has no unit):",
        f"  at the declared values: {sse_declared:.6f} (all {num_segments} "
        "segments)" + (f", {sse_declared_used:.6f}{used}" if part else ""),
        f"  at the fitted values:   {sse_fitted:.6f}{used}  {lower}",
        *evidence,
        "",
        "Fitted values (declared -> fitted, and what the recordings say):",
    ]
    loose: List[str] = []
    for name in sorted(fitted):
        info = constraints.get(name, {})
        lines.extend(
            _parameter_lines(name, declared[name], fitted[name],
                             runs.get(name, fitted[name]), info, num_used))
        if info.get("verdict") in (Verdict.NOT_IDENTIFIED, Verdict.INSENSITIVE,
                                   Verdict.ANCHORED):
            loose.append(name)
    if loose:
        lines += [
            "",
            f"Not constrained by the recordings: {', '.join(loose)}. To "
            "measure one, collect a recording in which it changes a scored "
            "feature, or take it out of AGENT_PARAM_SPECS.",
        ]
    lines.append("")
    if exploratory:
        lines.append(
            "EXPLORATORY subset fit: nothing was applied or recorded, and "
            "the model keeps the values of its last canonical fit. Values "
            "that differ between subsets that each fit well show that the "
            "recordings differ in something the model does not capture, "
            "not a parameter value.")
    else:
        lines.append(
            "Applied: sim.run and sim.refine now roll the model out at the "
            "fitted value of every parameter, except where a line above "
            "names the value the model runs.")
    return "\n".join(lines)
