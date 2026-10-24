"""The fit's note on parameters held at an end of their declared range."""
# pylint: disable=protected-access

from types import SimpleNamespace

import numpy as np

from predicators import utils
from predicators.agent_sdk.tools import ToolContext
from predicators.approaches.agent_continual_approach import \
    AgentContinualApproach
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec
from predicators.code_sim_learning.identifiability import BoundHit, Verdict, \
    bound_hits, bound_reminder, describe_bounds, identifiability_report


def _result(names, values):
    return FitResult(names=list(names),
                     samples=np.array([values], dtype=float),
                     log_probs=np.zeros(1),
                     jacobian=None,
                     noise_sigma=0.05,
                     prior_sigma=np.full(len(names), 0.5))


def test_a_value_on_an_end_of_its_range_is_named_with_that_end():
    """A balloon mass the data pushes past the top of its range and a box mass
    pushed below the bottom of a log-scale range are named, each with its end
    and range; a value inside its range is not."""
    specs = [
        ParamSpec("balloon_mass", 0.01, lo=0.003, hi=0.03),
        ParamSpec("box_mass", 0.25, lo=0.08, hi=0.8, scale="log"),
        ParamSpec("drag", 0.25, lo=0.0, hi=1.0),
    ]
    values = {"balloon_mass": 0.03, "box_mass": 0.08, "drag": 0.23}

    def sse_fn(params):
        return 1e4 * ((params["balloon_mass"] - 0.06)**2 +
                      (np.log(params["box_mass"]) - np.log(0.05))**2 +
                      (params["drag"] - 0.23)**2)

    result = _result(list(values), list(values.values()))
    report = identifiability_report(result, sse_fn, specs, num_explainable=5)
    assert report["balloon_mass"]["verdict"] is Verdict.AT_BOUND
    assert report["balloon_mass"]["bound_edge"] == "hi"
    assert report["box_mass"]["bound_edge"] == "lo"
    assert "bound_edge" not in report["drag"]
    hits = bound_hits(report, specs, values)
    assert hits == [
        BoundHit("balloon_mass", 0.03, "hi", 0.003, 0.03),
        BoundHit("box_mass", 0.08, "lo", 0.08, 0.8),
    ]
    lines = describe_bounds(hits)
    assert lines[0].startswith("At a declared bound:")
    assert "balloon_mass" in lines[1]
    assert "0.03 at the upper end of [0.003, 0.03]" in lines[1]
    assert "0.08 at the lower end of [0.08, 0.8]" in lines[2]
    reminder = bound_reminder(hits)
    assert reminder is not None
    assert reminder.startswith(
        "The fit holds balloon_mass at the upper end "
        "(0.03) and box_mass at the lower end (0.08) of "
        "their declared ranges")


def test_a_value_inside_its_range_gives_no_note():
    """No parameter at an end: no lines and no reminder."""
    specs = [ParamSpec("drag", 0.25, lo=0.0, hi=1.0)]
    result = _result(["drag"], [0.4])
    report = identifiability_report(result,
                                    lambda p: 1e4 * (p["drag"] - 0.4)**2,
                                    specs,
                                    num_explainable=5)
    assert not bound_hits(report, specs, {"drag": 0.4})
    assert not describe_bounds([])
    assert bound_reminder([]) is None


def test_a_rehearsal_reads_the_bounds_of_the_fit_of_the_file_as_it_is(
        tmp_path):
    """The reminder a canonical fit publishes is served while ``simulator.py``
    is the file that fit ran on, and dropped after an edit."""
    utils.reset_config({})
    approach = object.__new__(AgentContinualApproach)
    approach._fitted_params = {}
    approach._tool_context = ToolContext()
    simulator_file = str(tmp_path / "simulator.py")
    with open(simulator_file, "w", encoding="utf-8") as f:
        f.write("RESIDUAL_FEATURES = {'box': ['z']}\n")
    setattr(approach, "_resolve_synthesis_paths",
            lambda: SimpleNamespace(simulator_file=simulator_file))
    approach._publish_probe_fit({"k": 1.0},
                                "cycle_000_vers_001",
                                simulator_file,
                                reach="REACH",
                                bounds="BOUNDS")
    assert approach._bound_note() == "BOUNDS"
    assert approach._reach_note() == "REACH"
    with open(simulator_file, "a", encoding="utf-8") as f:
        f.write("# edited\n")
    assert approach._bound_note() is None
    approach._publish_probe_fit({"k": 1.0}, "cycle_000_vers_002",
                                simulator_file)
    assert approach._bound_note() is None
