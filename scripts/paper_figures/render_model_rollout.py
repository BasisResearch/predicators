"""Render a recorded plan inside the agent's own learned simulator.

Some runs checked a plan in their model with ``render=False``, so they
left no image of what the model predicted. This rebuilds that check
offline for figure panels: it loads the run's launch flags, executes the
saved simulator version the agent had at the time, deploys the parameter
values the harness would have used, reconstructs the noisy observations
and belief the agent saw at the plan's start, and runs the same plan
through the same probe with rendering on.

Run on a compute node. The output is an illustration of the agent's
model, not a benchmark rerun and not an execution gate.
"""
import argparse
import json
import pickle
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.scene import render_pybullet_image
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.approaches.synthesis_validation import carry_over_params
from predicators.code_sim_learning.base_simulator import base_simulator_class
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


def load_run_config(run: Path) -> None:
    """Restore flags from a trusted local run's recorded launch command.

    Kept here rather than imported so the script also runs on the older
    checkouts that recorded runs used.
    """
    info = re.sub(r"\x1b\[[0-9;]*m", "", (run / "info.log").read_text())
    command = next(
        line.split("Running command: ", 1)[1] for line in info.splitlines()
        if "Running command:" in line)
    sys.argv = shlex.split(command)[1:]
    utils.reset_config(utils.parse_args())


def main() -> None:
    """Rebuild the belief at ``--step`` and render ``--plan`` in the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--level", default="L02")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step",
                        type=int,
                        required=True,
                        help="Level step at which the plan starts")
    parser.add_argument("--candidate",
                        type=Path,
                        required=True,
                        help="Trusted saved agent-written simulator version")
    parser.add_argument("--plan",
                        type=Path,
                        required=True,
                        help="Text file with the agent's plan, one skill "
                        "per line")
    parser.add_argument("--previous-params",
                        type=json.loads,
                        default={},
                        help="Values the harness had deployed before this "
                        "version, as JSON; carried where they fit the new "
                        "specs, as the harness does")
    parser.add_argument(
        "--mods",
        type=json.loads,
        default=None,
        help="Object feature overrides for the start state, "
        "as JSON, passed to the probe's reset as the agent did")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-states",
                        action="store_true",
                        help="Also pickle the rollout's states, so a later "
                        "step can re-render them in another scene layout")
    parser.add_argument("--show-memory-glue",
                        action="store_true",
                        help="Bridge: also render the final state with the "
                        "model's remembered glue drawn as patches")
    args = parser.parse_args()
    load_run_config(args.run)
    real: Any = create_new_env(CFG.env, do_cache=False)
    namespace: Dict[str, Any] = {
        "BaseSimulator": base_simulator_class(CFG.env),
        "np": np,
        "ParamSpec": ParamSpec
    }
    exec(args.candidate.read_text(), namespace)  # pylint: disable=exec-used
    cls = namespace["RESIDUAL_ENV"]
    model = cls(use_gui=False)
    # Rule and physical parameters follow the same carry rule.
    params = carry_over_params(
        args.previous_params,
        list(getattr(cls, "AGENT_PARAM_SPECS", [])) +
        list(getattr(cls, "PHYSICAL_PARAM_SPECS", [])))
    model.apply_physical_param_overrides(params)
    options = get_gt_options(CFG.env, skill_library="composite")
    try:
        with (args.run / args.level / "episodes.pkl").open("rb") as stream:
            episode = pickle.load(stream)[
                args.episode]  # Trusted local record.
        # The harness keys each observation's noise by level index (L01 is
        # 0), episode and step, so the same draws are reproduced here.
        level_index = int(args.level.lstrip("L")) - 1
        states = episode["states"]
        actions = restore_actions(episode["actions"], states[0],
                                  sorted(options))
        noise = ObservationNoise.from_cfg()
        # Programs without memory have no tracker and nothing to replay.
        tracker = make_subclass_latent_tracker(cls, lambda: dict(params))
        frames = []
        for index, state in enumerate(states[:args.step + 1]):
            observed = noise.perturb(
                state, step_rng(CFG.seed, level_index, args.episode, index))
            frames.append(observed)
            if tracker is not None:
                tracker.attach(observed,
                               None if index == 0 else actions[index - 1])
        assert tracker is None or not tracker.failed
        belief = smooth_frames(frames[-CFG.continual_belief_window:], noise,
                               CFG.continual_belief_window,
                               CFG.continual_belief_sigmas)
        current = belief.frame.copy()
        if tracker is not None:
            current.latent = tracker.latent
        task = Task(current, set())
        option_model = _OracleOptionModel(options, model.simulate)
        option_model.sim_env = model
        args.out.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(types=real.types,
                          predicates=real.predicates,
                          processes=set(),
                          options=options,
                          train_tasks=[task],
                          example_state=current,
                          current_task=task,
                          current_observation=current,
                          option_model=option_model)
        ctx.image_save_dir = str(args.out)
        # Probe renders draw ctx.env, which must be the model, not reality.
        ctx.env = model
        ctx.probe_option_model_provider = lambda: option_model
        approach = object.__new__(AgentSimLearningApproach)
        approach._base_env = model  # pylint: disable=protected-access
        approach._residual_env_cls = cls  # pylint: disable=protected-access
        approach._identified_physical_params = {}  # pylint: disable=protected-access
        approach._tool_context = ctx  # pylint: disable=protected-access
        # Older checkouts predate fresh-env validation scopes.
        scope = getattr(approach, "_fresh_candidate_validation_scope", None)
        if scope is not None:
            ctx.probe_validation_env_scope = scope
        plan = args.plan.read_text().strip()
        probe = BeliefProbe(ctx).reset(current=True, mods=args.mods)
        result = probe.run(plan, render=True)
        final = probe._state  # pylint: disable=protected-access
        assert final is not None
        memory = getattr(final, "latent", None) or {}
        glue = {
            f"{name}.glue_{face}": round(level, 3)
            for name, faces in memory.get("glue", {}).items()
            for face, level in faces.items() if level > 0
        }
        if args.show_memory_glue:
            # The model keeps glue in its memory, not in block features, so
            # the engine draws no patches. Mark every face the model has put
            # glue on for this one render; the engine's own patch threshold
            # is a latch level the model does not share. The rollout itself
            # is unchanged.
            shown = final.copy()
            for obj in shown:
                for face, level in memory.get("glue", {}).get(obj.name,
                                                              {}).items():
                    if f"glue_{face}" in obj.type.feature_names:
                        shown.set(obj, f"glue_{face}", float(level > 0))
            render_pybullet_image(ctx, "model_final_memory_glue", state=shown)
        states = list(getattr(result, "states", [])) or [final]
        # The probe's final state carries the model's memory (latent);
        # keep that copy last so later renders can show what it holds.
        states[-1] = final
        if args.save_states:
            with (args.out / "rollout_states.pkl").open("wb") as stream:
                pickle.dump(states, stream)
        # Index of each plan step's last state in the saved list.
        step_ends, total = [], 0
        for step in getattr(result, "steps", []) or []:
            total += int(step.get("num_actions", 0))
            step_ends.append(total)
        summary = dict(run=str(args.run),
                       level=args.level,
                       step=args.step,
                       mods=args.mods,
                       candidate=args.candidate.name,
                       parameters=params,
                       plan=plan,
                       report=str(result)[:4000],
                       remembered_glue=glue,
                       saved_states=len(states),
                       step_end_indices=step_ends,
                       welds=memory.get("welds", []),
                       images=sorted(p.name for p in args.out.glob("*.png")))
        (args.out /
         "rollout.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2)[:3000], flush=True)
    finally:
        model.dispose()
        real.dispose()


if __name__ == "__main__":
    main()
