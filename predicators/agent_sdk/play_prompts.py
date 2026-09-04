"""Prompt builders for continual-protocol play sessions.

Templates: ``prompts/play_system.md`` (one prompt per run) and
``prompts/play_query.md`` (one query per session). Both are rendered
through :mod:`prompt_templates` so the call sites and the templates
cannot drift.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from predicators.agent_sdk.prompt_templates import render

# Tool descriptions shown in the system prompt, in this order. The
# tool schemas carry the argument details; this list is the map.
TOOL_BLURBS = {
    "env_observe":
    "the current observation: episode state, goal, environment atoms, "
    "your predicates, object features, a render, the ledger. Free.",
    "env_step":
    "one primitive action (a low-level action vector). One step.",
    "env_reset":
    "restart the current level from its initial state. Charged the "
    "reset price in steps and one reset. The only valid action after "
    "GAME_OVER on a level with resets.",
    "env_end_run":
    "end the run for this environment (takes effect when the session "
    "ends). Forfeits every remaining level; a last resort.",
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
    "code in the sandbox with the `sim` belief probe. Free.",
    "learn_run":
    "queue a learning session over every recorded episode (simulator "
    "synthesis, parameter fit, predicate invention); runs after this "
    "session ends. Free in steps.",
    "session_end":
    "end this session with a handoff note for the next one.",
}


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
                             reset_cost: int = 1) -> str:
    """The system prompt of every play session; ``reset_cost`` is the steps one
    reset is charged (``continual_reset_cost``).

    The tool surface selects the variant: an arm with ``run_python`` has
    a belief model behind ``sim`` and one with ``learn_run`` can learn
    it; the model-free arm has neither, and its prompt says so instead
    of describing tools it does not have.
    """
    names = set(tool_names)
    model = "run_python" in names
    variant = "" if model else "_model_free"
    sections = [
        render("play_system", "identity" + variant),
        render("play_system", "protocol", reset_cost=str(int(reset_cost))),
        render("play_system", "tools", tool_list=render_tool_list(tool_names)),
        render("play_system", "grammar"),
        render("play_system",
               "sandbox",
               model_files=render("play_system",
                                  "sandbox" + variant + "_files")),
    ]
    if "learn_run" in names:
        sections.append(render("play_system", "learning"))
    sections += [
        render("play_system", "journal"),
        render("play_system",
               "session",
               learn_clause=(", when you want a learning session"
                             if "learn_run" in names else "")),
        render("play_system", "principles" + variant),
    ]
    return "\n\n".join(sections)


def render_learning_status(*, n_learn: int, sim_version: Optional[str],
                           pred_version: Optional[str], fit_status: str,
                           n_episodes: int, n_steps: int,
                           n_new_episodes: int) -> str:
    """The learning-status block of the query."""
    if n_learn == 0:
        return render("play_query",
                      "learning_none",
                      n_episodes=str(n_episodes),
                      n_steps=str(n_steps))
    return render("play_query",
                  "learning_some",
                  n_learn=str(n_learn),
                  sim_version=sim_version or "none",
                  pred_version=pred_version or "none",
                  fit_status=fit_status or "unknown",
                  n_episodes=str(n_episodes),
                  n_steps=str(n_steps),
                  n_new=str(n_new_episodes))


def render_data_status(*, n_episodes: int, n_steps: int) -> str:
    """The learning-status block of an arm without a belief model."""
    return render("play_query",
                  "learning_model_free",
                  n_episodes=str(n_episodes),
                  n_steps=str(n_steps))


def build_play_query(*, session_number: int, resumed: bool, level_number: int,
                     levels_total: int, goal_nl: str,
                     goal_atoms: Sequence[str], ledger: str, observation: str,
                     skills: str, predicates: str, types: str, learning: str,
                     journal: str, attempts: str, handoff: str) -> str:
    """The query that opens one play session."""
    if resumed:
        opening = render("play_query", "opening_resumed")
    elif session_number <= 1:
        opening = render("play_query", "opening_first")
    else:
        opening = render("play_query",
                         "opening_next",
                         session_number=str(session_number))
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
        observation=observation,
        skills=skills,
        predicates=predicates,
        types=types,
        learning=learning,
        journal=journal or render("play_query", "no_journal"),
        attempts=attempts or render("play_query", "no_attempts"),
        handoff=handoff or render("play_query", "no_handoff"),
        instructions=render("play_query", "instructions"),
    )
