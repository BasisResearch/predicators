"""The screen of the engine materials a model does not declare.

Under ``belief_keep_undeclared_materials`` every joint draw of a
rehearsal holds such a material at the scene's own value, so P-hat says
nothing about it. A scene value that is wrong, in a material the plan
turns on, then gives a rehearsal that is confident and wrong.

Under ``belief_undeclared_material_screen`` the harness reruns a plan
that a rehearsal let through: once at the scene's values, and once at
each end of each undeclared material's plausible range, every other
value at the fit. The runs share the start state and the planner seed,
so two of them differ in one material alone. Where the plan's outcome
at an end differs from its outcome at the scene's value, the screen
names the material, and the agent declares it: the fit then measures
it, and the draws carry what the recordings leave open.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from predicators.code_sim_learning.fit_space import ParamSpec

# The materials a note describes one by one; it lists the names of the
# rest.
_MAX_DESCRIBED = 8


def screen_points(specs: Sequence[ParamSpec]) -> List[Dict[str, float]]:
    """The points a screen of ``specs`` runs: the scene's own values first (no
    override), then each material at the low end and at the high end of its
    range, except an end the scene's value sits on."""
    points: List[Dict[str, float]] = [{}]
    for spec in specs:
        for end in (spec.lo, spec.hi):
            if end is None or math.isclose(
                    end, spec.init_value, rel_tol=1e-6, abs_tol=1e-12):
                continue
            points.append({spec.name: float(end)})
    return points


def outcome(run: Dict[str, Any]) -> str:
    """What one run of the plan came to: ``lost`` (the episode ended in
    GAME_OVER), ``won`` (the goal held), ``stalled`` (a skill failed) or
    ``open`` (the plan ran to its end short of the goal)."""
    if run.get("lost"):
        return "lost"
    if run.get("success"):
        return "won"
    if run.get("failure"):
        return "stalled"
    return "open"


def _describe(run: Dict[str, Any]) -> str:
    kind = outcome(run)
    if kind == "lost":
        line = run.get("lost_line")
        where = f" during line {line + 1}" if line is not None else ""
        return f"ends the episode in GAME_OVER{where}"
    if kind == "won":
        return "reaches the goal"
    if kind == "stalled":
        return f"stops on a failed skill ({run['failure']})"
    return "runs to its end without the goal and without a GAME_OVER"


def changed_materials(
    specs: Sequence[ParamSpec], points: Sequence[Dict[str, float]],
    runs: Sequence[Dict[str, Any]]
) -> List[Tuple[ParamSpec, List[Tuple[float, Dict[str, Any]]]]]:
    """Each material of ``specs`` with the (value, run) pairs whose outcome
    differs from the outcome at the scene's values, ``runs[0]``.

    ``runs`` answers ``points`` in order and may stop short of them (a
    time budget ended the runs); a point with no run counts for nothing.
    """
    if not runs:
        return []
    base = outcome(runs[0])
    by_name: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
    for point, run in zip(points[1:], runs[1:]):
        (name, value), = point.items()
        if outcome(run) != base:
            by_name.setdefault(name, []).append((value, run))
    return [(spec, by_name[spec.name]) for spec in specs
            if spec.name in by_name]


def screen_note(specs: Sequence[ParamSpec], points: Sequence[Dict[str, float]],
                runs: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The note of a screen, or None when no material changed the plan's
    outcome."""
    changed = changed_materials(specs, points, runs)
    if not changed:
        return None
    lines = []
    for spec, ends in changed[:_MAX_DESCRIBED]:
        found = "; ".join(f"at {value:.3g} the plan {_describe(run)}"
                          for value, run in ends)
        lines.append(f"- `{spec.name}` (scene value {spec.init_value:.3g}): "
                     f"{found}.")
    rest = [spec.name for spec, _ in changed[_MAX_DESCRIBED:]]
    if rest:
        lines.append("- the same holds for " +
                     ", ".join(f"`{name}`" for name in rest) + ".")
    tested = len({name for point in points[1:len(runs)] for name in point})
    return (
        f"The draws of this rehearsal hold the {len(specs)} materials that "
        "simulator.py does not declare at the scene's own values, so P-hat "
        "does not cover them. The harness ran the plan again from the same "
        f"start for {tested} of them, at each end of the material's "
        "plausible range and with every other value at the fit. At the "
        f"scene's values the plan {_describe(runs[0])}. The outcome changes "
        f"with {len(changed)} of them:\n" + "\n".join(lines) +
        "\nDeclare each of these in simulator.py. The fit then measures it "
        "from the recordings, and the draws carry what the recordings leave "
        "open.")


def log_screen(log_dir: Optional[str], plan_text: str,
               points: Sequence[Dict[str, float]], runs: Sequence[Dict[str,
                                                                       Any]],
               seconds: float, note: Optional[str]) -> None:
    """Append one screen to ``<log_dir>/material_screens.jsonl``."""
    if not log_dir:
        return
    record = {
        "time":
        time.time(),
        "plan":
        plan_text,
        "seconds":
        round(seconds, 1),
        "points":
        len(points),
        "runs": [{
            "params": dict(point),
            "outcome": outcome(run),
            "lost_line": run.get("lost_line"),
            "failure": run.get("failure"),
        } for point, run in zip(points, runs)],
        "note":
        note,
    }
    try:
        with open(os.path.join(log_dir, "material_screens.jsonl"),
                  "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logging.warning("Could not log the material screen: %s", e)
