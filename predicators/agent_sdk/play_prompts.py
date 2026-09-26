"""Prompt builders for continual-protocol play sessions.

Templates: ``prompts/play_system.md`` (one prompt per run) and
``prompts/play_query.md`` (one query per round of the run's
conversation). Both are rendered
through :mod:`prompt_templates` so the call sites and the templates
cannot drift.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

from predicators.agent_sdk.prompt_templates import render
from predicators.observation_noise import ObservationNoise
from predicators.settings import CFG

# Tool descriptions shown in the system prompt, in this order. The
# tool schemas carry the argument details; this list is the map.
TOOL_BLURBS = {
    "env_observe":
    "the current observation: episode state, goal, environment atoms, "
    "your predicates, object features, current joint_positions and their "
    "action-space order, a render, the ledger. Free.",
    "env_step":
    "one primitive action (a low-level action vector). One step.",
    "env_run_policy":
    "run your Python get_action(observation, memory) policy from a sandbox "
    "file for at most max_steps. Every returned action costs one step.",
    "env_reset":
    "restart the current level from its initial state. One step and "
    "one reset, and a last resort. The only valid action after "
    "GAME_OVER on a level with resets.",
    "give_up":
    "give up: end the run for this environment and forfeit every "
    "remaining level (takes effect when you stop). A last resort.",
    "skills_list":
    "the skill library: signatures, parameter meanings and ranges. Free.",
    "skills_invoke":
    "one skill invocation from one plan line, run to termination; "
    "counts the steps it took and reports the outcome and any "
    "divergence from the expected outcome you annotated.",
    "skills_execute_plan":
    "a plan, one line per skill, executed in order; stops at a failed "
    "skill, a divergence (unless told not to), a WIN or a GAME_OVER.",
    "run_python":
    "code in the sandbox with the `sim` probe over your model files "
    "(`sim.fit`, `sim.residuals`, `sim.run`, `sim.refine`, ...). Free.",
}
# The run_python blurb of the arms whose probe never fits (frozen
# dynamics, no harness fitting).
_RUN_PYTHON_NO_FIT_BLURB = (
    "code in the sandbox with the `sim` probe over your model files "
    "(`sim.residuals`, `sim.run`, `sim.refine`, ...). Free.")


def build_minimal_play_system_prompt(*, model_based: bool) -> str:
    """Instructions for the arms with no supplied skill library."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.tools.continual_tools import \
        PRIMITIVE_TOOL_NAMES

    variant = "" if model_based else "_model_free"
    sections = [
        render("play_minimal", "identity"),
        render("play_minimal", "protocol")
    ]
    noise = ObservationNoise.from_cfg()
    if noise.enabled and noise.declared:
        sections.append(
            render("play_system",
                   "observation_noise" + variant,
                   noise_line=noise.describe() + "."))
    sections.extend([
        render("play_system",
               "tools",
               tool_list=render_tool_list(PRIMITIVE_TOOL_NAMES)),
        render("play_minimal", "policy"),
        render("play_minimal", "sandbox"),
        render("play_minimal", "model_based" if model_based else "model_free"),
        render("play_system", "journal" + variant),
        render("play_system", "context"),
    ])
    return "\n\n".join(section.strip() for section in sections)


# Appended to the skill tools' blurbs for the model arm (the one with
# `sim`) under continual_skill_preflight.
_PREFLIGHT_BLURB = (" Rehearsed in `sim` from the last observation first; "
                    "a controller failure there refuses the request, "
                    "charging nothing, and `force=true` skips the "
                    "rehearsal.")


def render_tool_list(tool_names: Iterable[str],
                     fit_available: bool = True) -> str:
    """One bullet per tool the session exposes.

    ``fit_available`` False drops ``sim.fit`` from the ``run_python``
    blurb, for the arms whose probe refuses to fit.
    """
    names = list(tool_names)
    preflight = "run_python" in names and CFG.continual_skill_preflight
    lines = []
    for name in names:
        blurb = TOOL_BLURBS.get(name)
        if blurb is None:
            continue
        if name == "run_python" and not fit_available:
            blurb = _RUN_PYTHON_NO_FIT_BLURB
        if preflight and name in ("skills_invoke", "skills_execute_plan"):
            blurb += _PREFLIGHT_BLURB
        lines.append(f"- `{name}`: {blurb}")
    return "\n".join(lines)


def build_play_system_prompt(tool_names: Sequence[str],
                             base_sim_refs: Sequence[str] = (),
                             model_contract: str = "",
                             frozen_section: str = "",
                             frozen_model_supplied: bool = False,
                             oracle_dynamics: bool = False,
                             scene_built: bool = False,
                             scene_package: bool = False) -> str:
    """The system prompt of the run's conversation.

    The tool surface selects the variant: an arm with ``run_python``
    keeps and uses a belief model in the sandbox (``sim``); the model-
    free arm has neither, and its prompt says so instead of describing
    tools it does not have. ``base_sim_refs`` are the read-only base-
    simulator source paths, listed for the model arm, and
    ``model_contract`` is the rendered contract of its model files
    (:func:`build_model_contract`), placed after the model section.

    Each comparison arm gets a prompt that describes only what it can
    do. ``frozen_section`` is the rendered arm statement of a frozen-
    dynamics arm (``play_frozen.md``): it opens the prompt, and the
    frozen workflow and workbench replace the learning ones, so no
    later section asks for fitting or repair. ``frozen_model_supplied``
    says the harness wrote the model (scene-only, oracle dynamics), so
    the workbench describes a fixed file; the zero-shot arm writes its
    own before sealing it. The point-estimate arm
    (``continual_uncertainty_decisions`` off) gets the workflow, repair
    and workbench variants without uncertainty sweeps, and the no-
    fitting arm (``agent_sim_learn_declared_params_only``) is told to
    declare values rather than fit them. ``scene_built`` (the agentic
    real-to-sim arm) says no domain twin backs ``sim``: the agent builds
    the scene from the engine, the manifest and the assets, so the
    identity, the arm statement, the workflow's first-round line, the
    workbench's before-model line and the reference listing describe
    that. ``scene_package`` (a twin-backed arm that also receives the
    engine, the manifest and the assets) describes that reference
    listing instead; for the model-free arm it adds that listing as
    plain files to use however the agent likes, with no simulator.
    """
    names = set(tool_names)
    model = "run_python" in names
    variant = "" if model else "_model_free"
    frozen = model and bool(frozen_section)
    supplied = frozen and frozen_model_supplied
    point_estimate = model and not CFG.continual_uncertainty_decisions
    raw_point_estimate = (point_estimate
                          and not CFG.agent_sim_learn_declared_params_only
                          and not CFG.continual_belief_frame
                          and not CFG.code_sim_learning_rollout_noise_filter)
    fit_available = (model and not frozen
                     and not CFG.agent_sim_learn_declared_params_only)
    identity = ("identity_frozen" if supplied else
                "identity_real_to_sim" if scene_built else "identity" +
                variant)
    sections = [
        render("play_system", identity),
        render("play_system", "protocol"),
        render("play_system", "observations" + variant),
    ]
    noise = ObservationNoise.from_cfg()
    if noise.enabled and noise.declared:
        sections.append(
            render("play_system",
                   "observation_noise_raw"
                   if raw_point_estimate else "observation_noise" + variant,
                   noise_line=noise.describe() + "."))
    if frozen:
        sections.append(frozen_section)
    if scene_built:
        sections.append(
            render("play_system",
                   "arm_from_assets" if fit_available else "arm_real_to_sim"))
    # The joint belief replaces trials / solved / belief_draws with one
    # rehearsal over its K joint draws (paper Section 3.3).
    joint = (int(CFG.belief_joint_draws) > 0 and not point_estimate
             and bool(CFG.continual_uncertainty_decisions))
    final_rehearsal = render(
        "play_system",
        "final_rehearsal_joint" if joint else "final_rehearsal_legacy")
    if joint:
        candidates = str(int(CFG.belief_refine_candidates))
        refine_cell = render("play_system",
                             "refine_cell_joint",
                             refine_candidates=candidates)
    else:
        refine_cell = render("play_system", "refine_cell_legacy")
    adaptive = ""
    if (model and not frozen and not point_estimate
            and CFG.agent_explorer_info_seeking
            and CFG.agent_explorer_info_seeking_adaptive
            and not CFG.agent_model_repair):
        adaptive = render("play_system", "adaptive_info_seeking")
    if frozen:
        sections.append(
            render("play_system",
                   "workflow_frozen",
                   final_rehearsal=final_rehearsal))
    elif point_estimate:
        declared = CFG.agent_sim_learn_declared_params_only
        sections.append(
            render(
                "play_system",
                "workflow_point_estimate",
                model_ready=render(
                    "play_system", "model_ready_declared"
                    if declared else "model_ready_fitted"),
                sim_first_round=render(
                    "play_system", "sim_first_round_scene"
                    if scene_built else "sim_first_round_twin"),
                rehearse_line=render(
                    "play_system",
                    "rehearse_declared" if declared else "rehearse_fitted")))
    elif model:
        ready = ("model_ready_declared"
                 if CFG.agent_sim_learn_declared_params_only else
                 "model_ready_fitted")
        sections.append(
            render("play_system",
                   "workflow",
                   adaptive_info_seeking=adaptive,
                   final_rehearsal=final_rehearsal,
                   rehearsal_clause="once your scene model is loaded"
                   if scene_built else "model or not",
                   sim_first_round=render(
                       "play_system", "sim_first_round_scene"
                       if scene_built else "sim_first_round_twin"),
                   model_ready=render("play_system", ready)))
    if model:
        if not point_estimate and (not frozen or oracle_dynamics):
            sections.append(
                render("play_system",
                       "rehearsal_reliability",
                       state_rehearsal=render(
                           "play_system", "state_rehearsal_joint"
                           if joint else "state_rehearsal_legacy")))
        if oracle_dynamics:
            sections.append(render("play_system", "oracle_discrepancies"))
        if CFG.continual_require_model_on_test:
            sections.append(render("play_system", "model_gate"))
        if CFG.continual_skill_preflight:
            sections.append(render("play_system", "skill_preflight"))
        if CFG.agent_model_repair and not frozen:
            if point_estimate:
                sections.append(
                    render("play_system",
                           "model_repair_point_estimate",
                           repair_reference="the engine"
                           if scene_built else "the visible base",
                           repair_values="declared"
                           if CFG.agent_sim_learn_declared_params_only else
                           "fitted"))
            else:
                sections.append(
                    render("play_system",
                           "model_repair",
                           repair_reference=
                           "your reconstructed scene and the engine"
                           if scene_built else "the visible base"))
    else:
        sections.append(render("play_system", "workflow_model_free"))
    files = ("sandbox_frozen_files" if supplied else "sandbox" + variant +
             "_files")
    sections += [
        render("play_system",
               "tools",
               tool_list=render_tool_list(tool_names,
                                          fit_available=fit_available)),
        render("play_system", "grammar"),
        render("play_system",
               "sandbox",
               model_files=render("play_system", files)),
        render("play_system", "journal" + variant),
        render("play_system", "context"),
    ]
    if not model and scene_package and base_sim_refs:
        sections.append(
            render("play_system",
                   "scene_refs_model_free",
                   ref_listing="\n".join(f"- `{r}`" for r in base_sim_refs)))
    if model:
        refs = ("" if not base_sim_refs else render(
            "play_system",
            "scene_refs" if scene_built else
            "twin_scene_refs" if scene_package else "base_sim_refs",
            ref_listing="\n".join(f"- `{r}`" for r in base_sim_refs)))
        robustness = render(
            "play_system", "robustness_point_estimate" if point_estimate else
            "robustness_joint" if joint else "robustness_uncertainty")
        # The probe refuses physics_sweep without uncertainty decisions.
        if point_estimate:
            sweep_verdict = ""
        else:
            sweep_verdict = render(
                "play_system", "sweep_verdict_identified"
                if fit_available else "sweep_verdict_declared")
        if frozen:
            if oracle_dynamics:
                robustness = render(
                    "play_system", "robustness_oracle_joint"
                    if joint else "robustness_oracle")
                sweep_verdict = ""
            fixed = render(
                "play_system", "frozen_line_supplied"
                if frozen_model_supplied else "frozen_line_written")
            sections.append(
                render("play_system",
                       "model_frozen",
                       fixed_line=fixed,
                       refine_cell=refine_cell,
                       robustness_row=robustness,
                       sweep_verdict_line=sweep_verdict,
                       base_sim_refs=refs))
        else:
            fit_variant = "fitted" if fit_available else "declared"
            sections.append(
                render("play_system",
                       "model",
                       before_model_line=render(
                           "play_system", "before_model_scene"
                           if scene_built else "before_model_twin"),
                       after_edit_line=render("play_system",
                                              "after_edit_" + fit_variant),
                       fit_rows=render(
                           "play_system", "fit_rows_harness"
                           if fit_available else "fit_rows_declared"),
                       refine_cell=refine_cell,
                       robustness_row=robustness,
                       sweep_verdict_line=sweep_verdict,
                       base_sim_refs=refs))
        if model_contract:
            sections.append(model_contract)
    if point_estimate:
        sections.append(
            render(
                "play_system", "point_estimate_decisions_raw"
                if raw_point_estimate else "point_estimate_decisions_declared"
                if CFG.agent_sim_learn_declared_params_only else
                "point_estimate_decisions"))
    return "\n\n".join(section.strip() for section in sections)


def build_model_contract(
    *,
    partially_observable: bool,
    physical_params_section: str = "",
    declared_params_only: bool = False,
    frozen: bool = False,
    supplied_model: bool = False,
    scene_built: bool = False,
) -> str:
    """The contract of the model files, for the model arm's system prompt
    (``play_model_contract.md``).

    ``partially_observable`` adds the model-state callback contract and
    the latent-aware classifier note. ``physical_params_section`` is the
    rendered system-identification section, from
    ``render_physical_params_section`` in the learn prompt module; empty
    when the env reveals no tunable physics. ``declared_params_only``
    adds the no-harness-fitting section, since the probe then refuses to
    fit. ``frozen`` (the zero-shot arm) drops the fitting guidance,
    since the model is sealed at the first action. ``supplied_model``
    (scene-only, oracle dynamics) keeps only the predicate contract: the
    agent never writes ``simulator.py``. ``scene_built`` (the agentic
    real-to-sim arm) replaces the domain-twin subclass contract with the
    ``SceneBase`` one: the agent loads the scene itself.
    """
    if supplied_model:
        parts = [
            render("play_model_contract", "intro_supplied"),
            render("play_model_contract", "predicates"),
        ]
        if partially_observable:
            parts.append(render("play_model_contract", "predicates_latent"))
        return _join_contract(parts)
    parts = [
        render("play_model_contract", "intro"),
        render("subclass_model",
               "simulator_scene" if scene_built else "simulator"),
        render("subclass_model", "dynamics"),
    ]
    noise = ObservationNoise.from_cfg()
    if partially_observable:
        # With undeclared noise the prompt never mentions noise at all.
        parts.append(
            render("subclass_model",
                   "memory",
                   base_class="SceneBase" if scene_built else "BaseSimulator",
                   estimate_errors="errors in the model and noisy input"
                   if noise.declared else "errors in the model"))
    parts.append(render("play_model_contract", "paramspec"))
    if noise.enabled and noise.declared and not frozen:
        parts.append(
            render(
                "play_model_contract", "observation_noise_declared"
                if declared_params_only else "observation_noise"))
    if physical_params_section:
        parts.append(physical_params_section)
    if declared_params_only:
        # Without uncertainty decisions the declared ranges drive nothing.
        parts.append(
            render(
                "play_model_contract",
                "no_harness_fitting" if CFG.continual_uncertainty_decisions
                else "no_harness_fitting_point_estimate"))
    parts.append(render("play_model_contract", "predicates"))
    if partially_observable:
        parts.append(render("play_model_contract", "predicates_latent"))
    return _join_contract(parts)


def _join_contract(parts: List[str]) -> str:
    """Join contract sections, demoting every heading after the intro."""
    for index in range(1, len(parts)):
        parts[index] = parts[index].strip("\n")
        if parts[index].startswith("## "):
            parts[index] = "#" + parts[index]
    return "\n\n".join(parts)


def render_data_status(*, n_episodes: int, n_steps: int) -> str:
    """The learning-status block of an arm without a belief model."""
    return render("play_query",
                  "learning_model_free",
                  n_episodes=str(n_episodes),
                  n_steps=str(n_steps))


def build_play_query(*, kind: str, round_number: int, level_number: int,
                     levels_total: int, goal_nl: str,
                     goal_atoms: Sequence[str], ledger: str, context: str,
                     observation: str, skills: str, predicates: str,
                     types: str, model: str, journal: str,
                     attempts: str) -> str:
    """The message that opens one round of the run's conversation.

    ``kind`` is ``first`` (the run's first message), ``level`` (a new
    level in the same conversation), ``continue`` (the agent stopped
    before the level was settled) or ``resumed`` (after a preemption). A
    continuation is short: the conversation already holds the level.
    """
    assert kind in ("first", "level", "continue", "resumed"), kind
    if kind == "first":
        opening = render("play_query", "opening_first")
    else:
        opening = render("play_query",
                         "opening_" + kind,
                         round_number=str(round_number),
                         level_number=str(level_number))
    instructions = render("play_query", "instructions")
    if kind == "continue":
        return render("play_query",
                      "skeleton_continue",
                      opening=opening,
                      ledger=ledger,
                      context=context,
                      observation=observation,
                      model=model,
                      instructions=instructions)
    return render(
        "play_query",
        "skeleton",
        opening=opening,
        level_number=str(level_number),
        levels_total=str(levels_total),
        goal_nl=goal_nl or "(no description given; the atoms are the goal)",
        goal_atoms=", ".join(goal_atoms) if goal_atoms else
        "(not expressible in your predicates; the goal description above "
        "is the goal)",
        ledger=ledger,
        context=context,
        observation=observation,
        skills=skills,
        predicates=predicates,
        types=types,
        model=model,
        journal=journal or render("play_query", "no_journal"),
        attempts=attempts or render("play_query", "no_attempts"),
        instructions=instructions,
    )
