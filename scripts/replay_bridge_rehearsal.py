"""Rehearse a saved Bridge decision using only reconstructed observations.

Run on a compute node. This diagnoses reset fidelity and controller
outcomes; it does not enforce an execution gate or count as a benchmark
rerun.
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.tools.context import ToolContext
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning.base_simulator import base_simulator_class
from predicators.code_sim_learning.continual_oracle import oracle_source
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.latent_tracker import \
    make_subclass_latent_tracker
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.observation_belief import smooth_frames
from predicators.observation_noise import ObservationNoise, step_rng
from predicators.option_model import _OracleOptionModel
from predicators.run.recording import restore_actions
from predicators.settings import CFG
from predicators.structs import Task
from scripts.replay_oracle_certificate import load_run_config


def main() -> None:
    """Replay inferred memory, then run the recorded requested skill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--step", type=int, default=1267)
    parser.add_argument(
        "--candidate",
        type=Path,
        help="Trusted saved agent-written simulator to diagnose")
    parser.add_argument("--exact-observations",
                        action="store_true",
                        help="Offline diagnostic only; never used by an agent")
    args = parser.parse_args()
    load_run_config(args.run)
    assert CFG.env == "pybullet_bridge"
    real: Any = create_new_env(CFG.env, do_cache=False)
    namespace: Dict[str, Any] = {
        "BaseSimulator": base_simulator_class(CFG.env),
        "np": np,
        "ParamSpec": ParamSpec
    }
    source = args.candidate.read_text() if args.candidate else oracle_source()
    exec(source, namespace)  # pylint: disable=exec-used
    cls = namespace["RESIDUAL_ENV"]
    model = cls(use_gui=False)
    options = get_gt_options(CFG.env, skill_library="composite")
    try:
        with (args.run / "L02/episodes.pkl").open("rb") as stream:
            episode = pickle.load(stream)[0]
        states = episode["states"]
        actions = restore_actions(episode["actions"], states[0],
                                  sorted(options))
        noise = ObservationNoise.from_cfg()
        tracker = make_subclass_latent_tracker(cls, lambda: {})
        assert tracker is not None
        frames = []
        for index, state in enumerate(states[:args.step + 1]):
            observed = (state.copy() if args.exact_observations else
                        noise.perturb(state, step_rng(CFG.seed, 1, 0, index)))
            frames.append(observed)
            tracker.attach(observed,
                           None if index == 0 else actions[index - 1])
        assert not tracker.failed
        belief = smooth_frames(frames[-CFG.continual_belief_window:], noise,
                               CFG.continual_belief_window,
                               CFG.continual_belief_sigmas)
        current = (frames[-1].copy()
                   if args.exact_observations else belief.frame.copy())
        current.latent = tracker.latent
        model._set_state(current)  # pylint: disable=protected-access
        restored = model._get_state()  # pylint: disable=protected-access
        errors = sorted(
            (abs(current.get(o, f) - restored.get(o, f)), o.name, f)
            for o in current if o.type.name == "block"
            for f in ("x", "y", "z", "roll", "pitch", "yaw"))
        pose_error = errors[-1][0]
        print(
            json.dumps({
                "step":
                args.step,
                "candidate":
                str(args.candidate) if args.candidate else "oracle",
                "parameters":
                "declared defaults, not a recovered historical fit",
                "exact_observations":
                args.exact_observations,
                "max_restore_feature_error":
                pose_error,
                "largest_errors":
                errors[-6:],
                "welds":
                len(model._weld_constraint_edges()),  # pylint: disable=protected-access
                "held": [
                    o.name for o in restored
                    if "is_held" in o.type.feature_names
                    and restored.get(o, "is_held") > .5
                ]
            }),
            flush=True)
        # This is a controller diagnostic, not a task-solve check. Its
        # deliberately empty goal must not be reported as benchmark success.
        task = Task(current, set())
        option_model = _OracleOptionModel(options, model.simulate)
        option_model.sim_env = model
        ctx = ToolContext(types=real.types,
                          predicates=real.predicates,
                          processes=set(),
                          options=options,
                          train_tasks=[task],
                          example_state=current,
                          current_task=task,
                          current_observation=current,
                          option_model=option_model)
        ctx.probe_option_model_provider = lambda: option_model
        approach = object.__new__(AgentSimLearningApproach)
        approach._base_env = model  # pylint: disable=protected-access
        approach._residual_env_cls = cls  # pylint: disable=protected-access
        approach._identified_physical_params = {}  # pylint: disable=protected-access
        approach._tool_context = ctx  # pylint: disable=protected-access
        # pylint: disable-next=protected-access
        scope = approach._fresh_candidate_validation_scope
        ctx.probe_validation_env_scope = scope
        entries = [
            json.loads(line)
            for line in (args.run /
                         "L02/index.jsonl").read_text().splitlines()
        ]
        request = next(e for e in entries if e.get("start_step") == args.step
                       and e.get("skill", "").startswith("MoveTo"))
        robot = next(o for o in current if o.type.name == "robot")
        plan = f"MoveTo({robot}){request['params']}"
        result = BeliefProbe(ctx).reset(current=True).run(plan, trials=3)
        print(json.dumps({
            "plan":
            plan,
            "recorded_status":
            request["status"],
            "trials": [{k: v
                        for k, v in t.items() if k != "goal_reached"}
                       for t in result.trials],
            "fresh":
            result.fresh_env_per_trial
        }),
              flush=True)
        assert result.fresh_env_per_trial
    finally:
        model.dispose()
        real.dispose()


if __name__ == "__main__":
    main()
