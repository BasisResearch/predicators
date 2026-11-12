"""The text the arm with the supplied scene simulator gives its agent.

The system prompt, the query's status block and the sandbox CLAUDE.md of
the No learning arm. Its own sections are in
``prompts/play_scene_simulator.md``; the sections it shares with the
other arms (the run rules, the observations, the tools, the skill
grammar, the working files, the journal and the rounds) come from
``prompts/play_system.md`` as the other arms get them, less the few
phrases this arm words differently (:data:`_SHARED_WORDING`). A shared
passage that has changed fails the rendering, so a new wording cannot
reach this arm unseen.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from predicators.agent_sdk.play_prompts import TOOL_BLURBS
from predicators.agent_sdk.prompt_templates import render
from predicators.agent_sdk.sandbox_prompts import build_claude_md
from predicators.observation_noise import ObservationNoise
from predicators.settings import CFG

_ARM = "play_scene_simulator"
# The run_python line of the system prompt's tool list.
RUN_PYTHON_BLURB = (
    "code in the sandbox with `sim`, the simulator of the "
    "scene (`sim.run`, `sim.refine`, `sim.belief`, ...). Free.")
# The heading of the query's status block, shared and this arm's.
_QUERY_HEADING = ("## Model and data", "## Simulator and data")
# Shared passages this arm words differently: (template, section) -> the
# passage and its wording here.
_SHARED_WORDING: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
    ("play_system", "protocol"): [
        ("`WIN`: record what you learned and stop your response",
         "`WIN`: record your notes and stop your response"),
    ],
    ("play_system", "journal_model_free"): [
        ("Update `./journal.md` when you learn something, not only",
         "Update `./journal.md` when you find something out, not only"),
    ],
}
# The same for the sandbox CLAUDE.md.
_CLAUDE_MD_WORDING = [
    ("The kind is `play` for continual rounds, or a phase such as `learn`, "
     "`test`, or `explore`.", "The kind is `play`."),
]
_CLAUDE_MD_REFERENCES = ("Curated source files are in `./reference/`. Read "
                         "them to learn the APIs before writing code.")

# The entries of the scene manifest's legend that tell a reader how to
# build the scene in a model of its own, each as this arm's manifest has
# it: what the simulator does with what the entry describes.
_LEGEND_WORDING = {
    "omitted":
    ("the manifest lists shapes and joints; in the simulator a body loaded "
     "from a URDF file has the file's masses, a moving body built from "
     "shapes has the mass of its volume at 1000 kg/m^3, and the other "
     "material values are the engine's defaults where no file sets one"),
    "not_listed":
    ("the ground plane, the backdrop walls and the robot, which the "
     "simulator loads itself"),
    "joints":
    ("lower_limit and upper_limit are the limits the engine enforces, in "
     "the simulator too; where the deployment set other limits than its "
     "URDF file's, the joint also lists the file's as urdf_limits"),
    "hard_stop":
    ("a joint marked hard_stop: true never leaves its limits: an engine "
     "limit alone is soft, and an arm that keeps pushing drives the joint "
     "past it, so the deployment puts the joint back at its limit, at rest, "
     "after every physics step, and the simulator does the same"),
    "collision_margin":
    ("the contact margin the deployment builds a shape's link with, which "
     "the link has in the simulator too; the engine rounds a convex shape's "
     "edges by it, so where two flat surfaces meet edge to edge it can make "
     "a curb a slow body cannot climb"),
}


def scene_manifest_legend(legend: Mapping[str, str]) -> Dict[str, str]:
    """The legend of a scene manifest as this arm's reference files carry it:
    ``legend`` with the entries of :data:`_LEGEND_WORDING` in this arm's
    wording.

    An entry of the wording that ``legend`` no longer has is an error.
    """
    gone = sorted(set(_LEGEND_WORDING) - set(legend))
    if gone:
        raise ValueError(
            f"the scene manifest's legend has no entry {gone}; update "
            "_LEGEND_WORDING in scene_simulator_prompts.py to its current "
            "entries")
    return {
        key: _LEGEND_WORDING.get(key, text)
        for key, text in legend.items()
    }


def _reworded(text: str, wording: Sequence[Tuple[str, str]],
              where: str) -> str:
    """``text`` with each passage of ``wording`` replaced; a passage that does
    not occur exactly once is an error."""
    for old, new in wording:
        if text.count(old) != 1:
            raise ValueError(
                f"{where}: the passage {old!r} occurs {text.count(old)} "
                "times, not once; update its wording in "
                "scene_simulator_prompts.py to the current text")
        text = text.replace(old, new)
    return text


def _shared(section: str, **values: str) -> str:
    """A section of ``play_system.md`` in this arm's wording."""
    text = render("play_system", section, **values)
    wording = _SHARED_WORDING.get(("play_system", section), [])
    return _reworded(text, wording, f"play_system.md#{section}")


def _tool_list(tool_names: Sequence[str]) -> str:
    """One bullet per tool of the session, as the shared list words them, with
    this arm's line for ``run_python``."""
    lines = []
    for name in tool_names:
        blurb = RUN_PYTHON_BLURB if name == "run_python" else \
            TOOL_BLURBS.get(name)
        if blurb is not None:
            lines.append(f"- `{name}`: {blurb}")
    return "\n".join(lines)


def build_scene_simulator_system_prompt(
    tool_names: Sequence[str],
    scene_refs: Sequence[str],
    source_refs: Sequence[str] = ()) -> str:
    """The system prompt of the arm with the supplied scene simulator.

    ``scene_refs`` are the agent-visible paths of the scene files, and
    ``source_refs`` those of the simulator's source, which the prompt
    lists for reference after the scene files when there are any. The
    gate, claim-check and predictive-check sections follow the flags
    that turn those switches on, as they do in the other arms' prompt.
    """
    sections = [
        render(_ARM, "identity"),
        _shared("protocol"),
        _shared("observations_model_free"),
    ]
    noise = ObservationNoise.from_cfg()
    if noise.enabled and noise.declared:
        sections.append(
            _shared("observation_noise", noise_line=noise.describe() + "."))
    sections += [render(_ARM, "simulator"), render(_ARM, "workflow")]
    if CFG.continual_rehearsal_gate:
        sections.append(render(_ARM, "rehearsal_gate"))
        if CFG.continual_rehearsal_win_first:
            sections.append(_shared("rehearsal_win_first"))
        if CFG.continual_rehearsal_safe_plan:
            sections.append(render(_ARM, "rehearsal_safe_plan"))
        if CFG.continual_rehearsal_still_start:
            sections.append(
                _shared("rehearsal_still_start",
                        wait=str(CFG.continual_rehearsal_still_wait)))
    if CFG.continual_practise_after_reset:
        sections.append(render(_ARM, "practise_after_reset"))
    if CFG.continual_claim_check:
        sections.append(
            render(_ARM,
                   "claim_check",
                   resets=str(CFG.continual_claim_check_resets),
                   win=f"{CFG.continual_claim_check_win_p_hat:.2f}",
                   low=f"{CFG.continual_claim_check_low_p_hat:.2f}"))
    if CFG.continual_predictive_check:
        sections.append(render(_ARM, "predictive_check"))
    sections += [
        _shared("tools", tool_list=_tool_list(tool_names)),
        _shared("grammar"),
        _shared("sandbox", model_files=render(_ARM, "files")),
        _shared("journal_model_free"),
        _shared("context"),
        render(_ARM,
               "workbench",
               refine_candidates=str(int(CFG.belief_refine_candidates)),
               num_draws=str(int(CFG.belief_joint_draws))),
        render(_ARM,
               "scene_refs",
               ref_listing="\n".join(f"- `{r}`" for r in scene_refs)),
    ]
    if source_refs:
        sections.append(
            render(_ARM,
                   "source_refs",
                   ref_listing="\n".join(f"- `{r}`" for r in source_refs)))
    return "\n\n".join(section.strip() for section in sections)


def scene_simulator_status(n_episodes: int, n_steps: int) -> str:
    """The query's status block: the simulator and the recorded data."""
    return render(_ARM,
                  "status",
                  n_episodes=str(n_episodes),
                  n_steps=str(n_steps))


def scene_simulator_query(query: str) -> str:
    """A round query with this arm's heading over its status block.

    The heading is the first of its kind in the query: the journal and
    the attempts record, which the agent's own words fill, come after
    it.
    """
    shared, own = _QUERY_HEADING
    if shared not in query:
        raise ValueError(f"the round query has no {shared!r} heading; update "
                         "_QUERY_HEADING to the query's current text")
    return query.replace(shared, own, 1)


def build_scene_simulator_claude_md(source: bool = False) -> str:
    """The sandbox CLAUDE.md of this arm: the shared one, with the reference
    files described as the scene files they are here, and as the simulator's
    source too when ``source``."""
    references = ("claude_md_references_source"
                  if source else "claude_md_references")
    wording = _CLAUDE_MD_WORDING + [
        (_CLAUDE_MD_REFERENCES, render(_ARM, references))
    ]
    return _reworded(build_claude_md(), wording, "sandbox_claude_md.md")
