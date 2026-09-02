"""Prompt construction for the learning (simulator synthesis) phase.

Rendered from ``learn_system.md``, ``learn_message.md``,
``learn_predicate_invention.md``, and ``learn_partial_observability.md``
in ``predicators/agent_sdk/prompts`` (see :mod:`prompt_templates`). The
approach classes gather the per-instance values (digests, data roster,
reports, paths) and call these pure builders, so every prompt can be
rendered and reviewed without a live session.
"""
import re
from typing import Any, Mapping, Sequence

from predicators.agent_sdk.prompt_templates import render

_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _join(parts: Sequence[str]) -> str:
    text = "\n\n".join(p.strip("\n") for p in parts if p and p.strip())
    return _BLANK_RUN_RE.sub("\n\n", text).strip("\n") + "\n"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def build_learn_system_prompt(
    *,
    partially_observable: bool,
    residual_rule_signature: str,
    scene_viz_hint: str,
    physical_params_section: str = "",
    extra_sections: Sequence[str] = (),
    latent_extra_sections: Sequence[str] = (),
    workflow_extra: str = "",
    declared_params_only: bool = False,
) -> str:
    """Compose the synthesis system prompt.

    ``partially_observable`` selects the recurrent 5-argument rule
    signature and appends the recurrent-rules tutorial;
    ``residual_rule_signature`` is the matching ``def`` line for the
    geometric-gate example. ``physical_params_section`` is the rendered
    system-identification section (empty when the env reveals no
    parameters). ``extra_sections`` (subclass additions such as
    predicate invention) are inserted after the validation guidance;
    ``latent_extra_sections`` follow the recurrent-rules tutorial (only
    rendered when ``partially_observable``); ``workflow_extra`` is
    appended to the workflow's validation step. ``declared_params_only``
    adds the no-estimation section (ablation A3): every parameter is
    used as declared, so the declaration is the estimate.
    """
    signature = render(
        "learn_system",
        "rule_signature_po" if partially_observable else "rule_signature_fo")
    parts = [
        render("learn_system", "intro"),
        render("learn_system", "produce"),
        physical_params_section,
        signature,
        render("learn_system", "cmds"),
        render("learn_system", "multi_object"),
        render("learn_system", "timing"),
        render("learn_system",
               "geometric_gates",
               residual_rule_signature=residual_rule_signature,
               scene_viz_hint=scene_viz_hint),
        render("learn_system", "paramspec"),
        render("learn_system", "declared_params")
        if declared_params_only else "",
        render("learn_system", "preinjected"),
        render("learn_system", "tools"),
        render("learn_system", "validation"),
        *extra_sections,
    ]
    if partially_observable:
        parts.append(render("learn_partial_observability", "rules"))
        parts.extend(latent_extra_sections)
    parts += [
        render("learn_system", "plan_format"),
        render("learn_system", "deliverables"),
        render("learn_system",
               "workflow",
               workflow_extra=(" " +
                               workflow_extra) if workflow_extra else ""),
    ]
    return _join(parts)


def render_physical_params_section(
        info: Mapping[str, Mapping[str, Any]]) -> str:
    """The ``PHYSICAL_PARAMS`` section for a revealed parameter menu.

    ``info`` maps a parameter name to its ``default``, ``lo``, ``hi``,
    ``description``, and optional ``scale``; empty input renders
    nothing, so envs without a menu never see the feature mentioned.
    """
    if not info:
        return ""
    lines = []
    for name, meta in info.items():
        scale_note = (", fitted in log-space"
                      if meta.get("scale") == "log" else "")
        lines.append(f"- `{name}` (built-in {meta['default']:.4g}, fit "
                     f"box [{meta['lo']:.4g}, {meta['hi']:.4g}]"
                     f"{scale_note}): {meta['description']}")
    return render("learn_system",
                  "physical_params",
                  param_list="\n".join(lines))


def render_predicate_invention_section(scene_workbench: str) -> str:
    """The predicate-invention system-prompt section."""
    return render("learn_predicate_invention",
                  "system",
                  scene_workbench=scene_workbench)


def render_predicate_latent_section() -> str:
    """The predicate-side latent guidance (invention arms, PO only)."""
    return render("learn_partial_observability", "predicates")


def render_predicate_workflow_extra() -> str:
    """The invention arm's addition to the workflow's validation step."""
    return render("learn_predicate_invention", "workflow_extra")


# ---------------------------------------------------------------------------
# First message
# ---------------------------------------------------------------------------


def build_learn_message(
        *,
        n_trajs: int,
        n_transitions: int,
        n_demos: int,
        n_interaction: int,
        trajectory_listing: str,
        structs_ref: str,
        inferred_hint: str,
        predicate_listing: str,
        types_digest: str,
        options_digest: str,
        simulator_file: str,
        objective_block: str = "",
        prior_state_block: str = "",
        divergence_block: str = "",
        base_sim_block: str = "",
        tools_block: str = "",
        extra_messages: Sequence[str] = (),
) -> str:
    """Compose the synthesis session's first message.

    Every block argument is already rendered (see the ``render_*``
    helpers below) or empty. ``extra_messages`` (predicate invention,
    partial observability, sampler synthesis) are appended in order.
    """
    body = render(
        "learn_message",
        "skeleton",
        n_trajs=str(n_trajs),
        n_transitions=str(n_transitions),
        n_demos=str(n_demos),
        n_interaction=str(n_interaction),
        trajectory_listing=trajectory_listing.strip("\n"),
        objective_block=objective_block,
        prior_state_block=prior_state_block,
        divergence_block=divergence_block,
        structs_ref=structs_ref,
        base_sim_block=base_sim_block,
        inferred_hint=inferred_hint,
        predicate_listing=predicate_listing,
        types_digest=types_digest.strip("\n"),
        options_digest=options_digest.strip("\n"),
        tools_block=tools_block,
        simulator_file=simulator_file,
    )
    return _join([body, *extra_messages])


def render_divergence_block(report: str, has_prior_model: bool) -> str:
    """The start-of-session residual report section."""
    return render("learn_message",
                  "divergence_prior" if has_prior_model else "divergence_base",
                  report=report.strip("\n"))


def render_base_sim_block(refs: Sequence[str]) -> str:
    """The base-simulator source listing, or empty."""
    if not refs:
        return ""
    return render("learn_message",
                  "base_sim",
                  ref_listing="\n".join(f"  - {r}" for r in refs))


def render_tools_block(tool_names: Sequence[str]) -> str:
    """The session's tool roster, or empty."""
    if not tool_names:
        return ""
    return render("learn_message",
                  "tools",
                  tool_listing="\n".join(f"  - {t}" for t in tool_names))


def render_objective_block(description: str) -> str:
    """The env's public task objective section, or empty."""
    if not description:
        return ""
    return render("learn_message", "objective", description=description)


def render_prior_state_block(prior_files: Sequence[str]) -> str:
    """The prior-cycle-state paragraph for the artifacts found, or empty."""
    if not prior_files:
        return ""
    return render("learn_message",
                  "prior_state",
                  prior_files=" and ".join(prior_files))


def render_predicate_invention_message(predicates_file: str,
                                       goal_block: str) -> str:
    """The invention arm's addition to the first message."""
    return render("learn_predicate_invention",
                  "message",
                  predicates_file=predicates_file,
                  goal_block=goal_block.strip("\n"))


def render_partial_observability_message() -> str:
    """The short partial-observability note for the first message."""
    return render("learn_partial_observability", "message")


def render_zero_shot_message() -> str:
    """The no-data note for the first message (ablation A1)."""
    return render("learn_message", "zero_shot")


# ---------------------------------------------------------------------------
# Program world model arm (C4)
# ---------------------------------------------------------------------------


def build_program_learn_system_prompt(
        *,
        scene_viz_hint: str,
        extra_sections: Sequence[str] = (),
        workflow_extra: str = "",
) -> str:
    """Compose the program-world-model synthesis system prompt.

    ``extra_sections`` (predicate invention) follow the validation
    guidance; the plan-format section is shared with the residual arm's
    template. ``scene_viz_hint`` is accepted for parity with the
    residual builder (the program template names the probe surface
    itself) and is not rendered.
    """
    del scene_viz_hint
    parts = [
        render("learn_program_system", "intro"),
        render("learn_program_system", "produce"),
        render("learn_program_system", "modeling"),
        render("learn_program_system", "tools"),
        render("learn_program_system", "validation"),
        *extra_sections,
        render("learn_system", "plan_format"),
        render("learn_program_system", "deliverables"),
        render("learn_program_system",
               "workflow",
               workflow_extra=(" " +
                               workflow_extra) if workflow_extra else ""),
    ]
    return _join(parts)


def build_program_learn_message(
        *,
        n_trajs: int,
        n_transitions: int,
        n_demos: int,
        n_interaction: int,
        trajectory_listing: str,
        structs_ref: str,
        predicate_listing: str,
        types_digest: str,
        options_digest: str,
        world_model_file: str,
        objective_block: str = "",
        prior_state_block: str = "",
        tools_block: str = "",
        extra_messages: Sequence[str] = (),
) -> str:
    """Compose the program-world-model synthesis session's first message."""
    body = render(
        "learn_program_message",
        "skeleton",
        n_trajs=str(n_trajs),
        n_transitions=str(n_transitions),
        n_demos=str(n_demos),
        n_interaction=str(n_interaction),
        trajectory_listing=trajectory_listing.strip("\n"),
        objective_block=objective_block,
        prior_state_block=prior_state_block,
        structs_ref=structs_ref,
        predicate_listing=predicate_listing,
        types_digest=types_digest.strip("\n"),
        options_digest=options_digest.strip("\n"),
        tools_block=tools_block,
        world_model_file=world_model_file,
    )
    return _join([body, *extra_messages])


def render_program_zero_shot_message() -> str:
    """The no-data note for the program arm's first message."""
    return render("learn_program_message", "zero_shot")

