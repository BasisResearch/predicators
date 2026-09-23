"""Audit recorded Domino outcomes with the current supplied model certificate.

Run on a compute node. The trusted local run directory supplies its
exact launch configuration and sanitized per-step recordings, never live
state. This is a certificate replay, not a new agent benchmark or a full
plan replay.
"""
import argparse
import json
import pickle
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

from predicators import utils
from predicators.agent_sdk.tools.verdicts import evaluate_states_with
from predicators.code_sim_learning.base_simulator import base_simulator_class
from predicators.code_sim_learning.continual_oracle import oracle_source
from predicators.envs import create_new_env
from predicators.envs.pybullet_domino.env import DominoEvaluator
from predicators.ground_truth_models import get_gt_options
from predicators.run.recording import restore_actions
from predicators.settings import CFG
from predicators.structs import GroundAtom


def load_run_config(run: Path) -> None:
    """Restore flags from a trusted local run's recorded launch command."""
    info = re.sub(r"\x1b\[[0-9;]*m", "", (run / "info.log").read_text())
    command = next(
        line.split("Running command: ", 1)[1] for line in info.splitlines()
        if "Running command:" in line)
    sys.argv = shlex.split(command)[1:]
    utils.reset_config(utils.parse_args())


def main() -> None:
    """Require acceptance on training and rejection on the failed test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    load_run_config(args.run)
    assert CFG.env == "pybullet_domino"
    real: Any = create_new_env(CFG.env, do_cache=False)
    namespace: Dict[str, Any] = {
        "BaseSimulator": base_simulator_class(CFG.env)
    }
    exec(oracle_source(), namespace)  # pylint: disable=exec-used
    model = namespace["RESIDUAL_ENV"](use_gui=False)
    options = sorted(get_gt_options(CFG.env, skill_library="composite"))
    scorecard = json.loads((args.run / "scorecard.json").read_text())
    try:
        for level, expected in (("L01", True), ("L02", False)):
            with (args.run / level / "episodes.pkl").open("rb") as stream:
                episodes = pickle.load(stream)
            episode = episodes[-1]
            states = episode["states"]
            actions = restore_actions(episode["actions"], states[0], options)
            recorded = scorecard["levels"][int(level[1:]) - 1]
            goal = set()
            for atom in recorded["goal"]:
                match = re.fullmatch(r"Toppled\(([^:]+):domino\)", atom)
                assert match is not None, atom
                obj = next(o for o in states[0] if o.name == match[1])
                pred = next(p for p in real.predicates if p.name == "Toppled")
                goal.add(GroundAtom(pred, [obj]))
            evaluator = DominoEvaluator(goal)
            labels = []
            for action in actions:
                option = action.get_option() if action.has_option() else None
                labels.append(None if option is None else (
                    option.name, tuple(o.name for o in option.objects),
                    tuple(float(v) for v in option.params)))
            verdict = evaluate_states_with(evaluator, states, labels, model)
            print(json.dumps({
                "level": level,
                "expected_solved": expected,
                **verdict
            }),
                  flush=True)
            assert verdict["terminated"], verdict
            assert verdict["solved"] == expected, verdict
            assert "fingertips-only" in verdict["note"], verdict
    finally:
        model.dispose()
        real.dispose()


if __name__ == "__main__":
    main()
