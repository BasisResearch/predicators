"""Compare push approaches at recorded decision points on compute nodes."""
# pylint: disable=protected-access
import argparse
import faulthandler
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.ground_truth_models.skill_factories import push
from predicators.settings import CFG
from predicators.structs import Action
from scripts.replay_oracle_certificate import load_run_config


def main() -> None:
    """Replay the original prefix, then execute a fresh grounded skill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--level", default="L02")
    parser.add_argument("--skills",
                        nargs="+",
                        default=["SwitchOn", "SwitchOff"])
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--start", type=int)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(90, repeat=True)
    load_run_config(args.run)
    if args.legacy:
        original = push.make_move_to_phase

        def legacy(*values: Any, **kwargs: Any) -> Any:
            kwargs["direct_descend"] = False
            kwargs["allow_approach_detour"] = False
            return original(*values, **kwargs)

        push.make_move_to_phase = legacy
    records = [
        json.loads(line)
        for line in (args.run / args.level /
                     "actions.jsonl").read_text().splitlines()
    ]
    records = [row for row in records if "a" in row and row["ep"] == 0]
    with (args.run / args.level / "episodes.pkl").open("rb") as stream:
        recorded_states = pickle.load(stream)[0]["states"]
    starts = [
        i for i, row in enumerate(records)
        if row.get("skill", {}).get("name") in args.skills and (
            i == 0 or row.get("skill") != records[i - 1].get("skill"))
    ]
    for start in starts[:args.max_cases]:
        if args.start is not None and start != args.start:
            continue
        record = records[start]["skill"]
        print("REPLAY_START " + json.dumps(record), flush=True)
        env = create_new_env(CFG.env, do_cache=False)
        result = {
            "run": str(args.run),
            "level": args.level,
            "start": start,
            "skill": record,
            "legacy": args.legacy
        }
        try:
            state = env.reset("train" if args.level == "L01" else "test", 0)
            for row in records[:start]:
                state = env.step(Action(np.asarray(row["a"],
                                                   dtype=np.float32)))
            result["prefix_joint_error"] = float(
                np.max(
                    np.abs(
                        np.asarray(state.joint_positions) -
                        np.asarray(recorded_states[start].joint_positions))))
            objects = {obj.name: obj for obj in state}
            option = next(opt for opt in get_gt_options(CFG.env)
                          if opt.name == record["name"])
            grounded = option.ground(
                [objects[name] for name in record["objects"]],
                np.asarray(record["params"], dtype=np.float32))
            assert grounded.initiable(state)
            steps = 0
            try:
                while not grounded.terminal(state) and steps < 500:
                    state = env.step(grounded.policy(state))
                    steps += 1
                result["status"] = "completed" if grounded.terminal(
                    state) else "step_limit"
            except Exception as error:  # pylint: disable=broad-except
                # Report failures, never treat them as passes.
                result["status"] = "error"
                result["error"] = str(error)
            result["steps"] = steps
            result["switches"] = {
                obj.name: state.get(obj, "is_on")
                for obj in state if "is_on" in obj.type.feature_names
            }
            print("REPLAY " + json.dumps(result), flush=True)
        finally:
            env.dispose()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
