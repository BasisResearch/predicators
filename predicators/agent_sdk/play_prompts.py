"""Prompt builders for continual-protocol play sessions.

Templates: ``prompts/play_system.md`` (one prompt per run) and
``prompts/play_query.md`` (one query per round of the run's
conversation). Both are rendered
through :mod:`prompt_templates` so the call sites and the templates
cannot drift.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from predicators.agent_sdk.prompt_templates import render
from predicators.observation_noise import ObservationNoise
from predicators.settings import CFG

# Tool descriptions shown in the system prompt, in this order. The
# tool schemas carry the argument details; this list is the map.
TOOL_BLURBS = {
    "env_observe":
    "the current observation: episode state, goal, environment atoms, "
    "your predicates, object features, a render, the ledger. Free.",
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


def build_minimal_play_system_prompt(*, model_based: bool) -> str:
    """Instructions for the arms with no supplied skill library."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.tools.continual_tools import \
        PRIMITIVE_TOOL_NAMES

    sections = [
        render("play_minimal", "identity"),
        render("play_minimal", "protocol")
    ]
    noise = ObservationNoise.from_cfg()
    if noise.enabled and noise.declared:
        sections.append(
            render("play_system",
                   "observation_noise",
                   noise_line=noise.describe() + "."))
    sections.extend([
        render("play_system",
               "tools",
               tool_list=render_tool_list(PRIMITIVE_TOOL_NAMES)),
        render("play_minimal", "policy"),
        render("play_minimal", "sandbox"),
        render("play_minimal", "model_based" if model_based else "model_free"),
        render("play_system", "journal"),
        render("play_system", "context"),
    ])
    return "\n\n".join(sections)


def render_tool_list(tool_names: Iterable[str]) -> str:
    """One bullet per tool the session exposes."""
    lines = []
    for name in tool_names:
        blurb = TOOL_BLURBS.get(name)
        if blurb is None:
            continue
        lines.append(f"- `{name}`: {blurb}")
    return "\n".join(lines)


def build_play_system_prompt(tool_names: Sequence[str],
                             base_sim_refs: Sequence[str] = (),
                             model_contract: str = "") -> str:
    """The system prompt of the run's conversation.

    The tool surface selects the variant: an arm with ``run_python``
    keeps and uses a belief model in the sandbox (``sim``); the model-
    free arm has neither, and its prompt says so instead of describing
    tools it does not have. ``base_sim_refs`` are the read-only base-
    simulator source paths, listed for the model arm, and
    ``model_contract`` is the rendered contract of its model files
    (:func:`build_model_contract`), placed after the model section.
    """
    names = set(tool_names)
    model = "run_python" in names
    variant = "" if model else "_model_free"
    sections = [
        render("play_system", "identity" + variant),
        render("play_system", "protocol"),
    ]
    noise = ObservationNoise.from_cfg()
    if noise.enabled and noise.declared:
        sections.append(
            render("play_system",
                   "observation_noise",
                   noise_line=noise.describe() + "."))
    sections += [
        render("play_system", "tools", tool_list=render_tool_list(tool_names)),
        render("play_system", "grammar"),
        render("play_system",
               "sandbox",
               model_files=render("play_system",
                                  "sandbox" + variant + "_files")),
    ]
    if model:
        refs = ("" if not base_sim_refs else render(
            "play_system",
            "base_sim_refs",
            ref_listing="\n".join(f"  - {r}" for r in base_sim_refs)))
        # Adaptive info-seeking: teach the submit-first protocol only when
        # the flag is on, so the always-on info-seeking arm (flag off) is
        # not told to hold probing back. A leading newline keeps the
        # placeholder line blank when empty.
        adaptive = ""
        if (CFG.agent_explorer_info_seeking
                and CFG.agent_explorer_info_seeking_adaptive
                and not CFG.agent_model_repair):
            adaptive = "\n" + render("play_system", "adaptive_info_seeking")
        sections.append(
            render("play_system",
                   "model",
                   base_sim_refs=refs,
                   adaptive_info_seeking=adaptive))
        if model_contract:
            sections.append(model_contract)
        if CFG.agent_model_repair:
            sections.append(render("play_system", "model_repair"))
    sections += [
        render("play_system", "journal"),
        render("play_system", "context"),
        render("play_system", "principles" + variant),
    ]
    return "\n\n".join(sections)


def build_model_contract(
    *,
    partially_observable: bool,
    physical_params_section: str = "",
    declared_params_only: bool = False,
) -> str:
    """The contract of the model files, for the model arm's system prompt
    (``play_model_contract.md``).

    ``partially_observable`` selects the recurrent rule signature and
    adds the hidden-state section and the latent-aware classifier note.
    ``physical_params_section`` is the rendered system-identification
    section, from ``render_physical_params_section`` in the learn prompt
    module; empty when the env reveals no tunable physics.
    ``declared_params_only`` adds the learn prompt's no-estimation
    section, since the probe then refuses to fit.
    """
    rule_args = ("state, latent, history, updates, params"
                 if partially_observable else "state, updates, params")
    latch_home = ("the `latent` block (see \"Hidden state\")"
                  if partially_observable else "a feature the rules own")
    parts = [
        render("play_model_contract", "intro"),
        render("play_model_contract", "simulator", rule_args=rule_args),
        render("play_model_contract",
               "processes",
               rule_args=rule_args,
               latch_home=latch_home),
        render("play_model_contract", "gates"),
    ]
    if partially_observable:
        parts.append(render("play_model_contract", "hidden_state"))
    parts.append(render("play_model_contract", "paramspec"))
    noise = ObservationNoise.from_cfg()
    if noise.enabled and noise.declared:
        parts.append(
            render("play_model_contract",
                   "observation_noise",
                   noise_line=noise.describe()))
    parts.append(render("play_model_contract", "subclass"))
    if physical_params_section:
        parts.append(physical_params_section)
    if declared_params_only:
        parts.append(render("learn_system", "declared_params"))
    parts.append(render("play_model_contract", "predicates"))
    if partially_observable:
        parts.append(render("play_model_contract", "predicates_latent"))
    return "\n\n".join(p.strip("\n") for p in parts)


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
